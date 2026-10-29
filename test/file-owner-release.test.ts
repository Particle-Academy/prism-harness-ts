import { createHash } from 'node:crypto';
import { lstat, mkdtemp, open, readFile, readdir, rename, rmdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, expect, it, vi } from 'vitest';
import { FileSessionStore } from '../src/index.js';

vi.mock('node:fs/promises', async importOriginal => {
  const actual = await importOriginal<typeof import('node:fs/promises')>();
  return {...actual, rename:vi.fn(actual.rename),rmdir:vi.fn(actual.rmdir)};
});
const actual=await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises');
it('exposes both concurrent pending releases as immutable store-scoped diagnostic entries',async()=>{
 const{store,directory}=await fixture(),other=await fixture();
 const keys=['private:pending:A','private:pending:B'];let callbacks=0,open!:()=>void;
 const barrier=new Promise<void>(done=>{open=done;});
 vi.mocked(rename).mockImplementation(async(source,target)=>{
  if(String(source).endsWith('.release.tmp'))throw Object.assign(Error('owned denial'),{code:'EPERM'});
  return actual.rename(source,target);
 });
 const results=await Promise.allSettled(keys.map(key=>store.withLock(key,async()=>{
  callbacks++;if(callbacks===2)open();await barrier;
 })));
 expect(results.every(result=>result.status==='rejected')).toBe(true);expect(callbacks).toBe(2);
 const rows=store.pendingReleases(),before=vi.mocked(rename).mock.calls.length;
 expect(rows).toHaveLength(2);expect(new FileSessionStore(directory).pendingReleases()).toEqual(rows);
 expect(other.store.pendingReleases()).toEqual([]);
 expect(rows.map(row=>row.key_digest).sort()).toEqual(keys.map(key=>createHash('sha256').update(key).digest('hex').slice(0,32)).sort());
 for(const row of rows){expect(Object.keys(row).sort()).toEqual(['key_digest','nonce']);expect(row.nonce).toMatch(/^[0-9a-f]{32}$/);expect(Object.isFrozen(row)).toBe(true);}
 expect(Object.isFrozen(rows)).toBe(true);
 expect(vi.mocked(rename).mock.calls.length).toBe(before);
 expect(JSON.stringify(rows)).not.toContain('private:pending:');
 expect(JSON.stringify(rows)).not.toContain(directory);
 vi.mocked(rename).mockImplementation(actual.rename);
 expect(await store.withLock('next-operation',()=>7)).toBe(7);
 expect(store.pendingReleases()).toEqual([]);
});
it('emits pending diagnostics with digest and nonce without raw keys or directory paths',async()=>{
 const{store,directory}=await fixture(),key='private:participant:do-not-log';
 const diagnostic=vi.spyOn(process,'emitWarning');
 vi.mocked(rename).mockImplementation(async(source,target)=>{
  if(String(source).endsWith('.release.tmp'))throw Object.assign(Error('owned denial'),{code:'EPERM'});
  return actual.rename(source,target);
 });
 try{
  await expect(store.withLock(key,()=>1)).rejects.toMatchObject({code:'EPERM'});
  const row=diagnostic.mock.calls.find(([,options])=>typeof options==='object'&&options?.code==='file_lock_release_pending');
  expect(row).toBeDefined();
  const message=String(row![0]),body=JSON.parse(message);
  expect(body).toEqual({key_digest:createHash('sha256').update(key).digest('hex').slice(0,32),nonce:expect.stringMatching(/^[0-9a-f]{32}$/)});
  expect(message).not.toContain(key);expect(message).not.toContain(directory);
 }finally{diagnostic.mockRestore();vi.mocked(rename).mockImplementation(actual.rename);}
 expect(await store.withLock('other-key',()=>7)).toBe(7);
});
afterEach(()=>{vi.mocked(rename).mockImplementation(actual.rename);vi.mocked(rmdir).mockImplementation(actual.rmdir);});
async function fixture(){
 const directory=await mkdtemp(join(tmpdir(),'prism-release-owner-'));
 return{store:new FileSessionStore(directory),directory,lock:join(directory,createHash('sha256').update('k').digest('hex').slice(0,32)+'.json.lock')};
}
it('retries the same closed release record after transient rename denial without callback replay',async()=>{
 const{store,directory}=await fixture(),records:Buffer[]=[];let callbacks=0;
 vi.mocked(rename).mockImplementation(async(source,target)=>{
  if(String(source).endsWith('.release.tmp')){
   records.push(await readFile(source));
   expect(JSON.parse(records.at(-1)!.toString('utf8')).released).toBe(true);
   if(records.length<3)throw Object.assign(Error('synthetic transient reader'),{code:'EPERM'});
  }
  return actual.rename(source,target);
 });
 expect(await store.withLock('k',()=>{callbacks++;return 7;})).toBe(7);
 expect(callbacks).toBe(1);expect(records).toHaveLength(3);
 expect(records[0]).toEqual(records[1]);expect(records[1]).toEqual(records[2]);
 expect(await readdir(directory)).toEqual([]);
});
it('keeps publish exhaustion observable and refuses to treat a release temporary as authority',async()=>{
 const{store,lock}=await fixture();let callbacks=0;
 vi.mocked(rename).mockImplementation(async(source,target)=>{
  if(String(source).endsWith('.release.tmp'))throw Object.assign(Error('synthetic persistent reader'),{code:'EPERM'});
  return actual.rename(source,target);
 });
 await expect(store.withLock('k',()=>{callbacks++;},-1)).rejects.toMatchObject({code:'EPERM'});
 expect(callbacks).toBe(1);
 const names=await readdir(lock);expect(names).toHaveLength(2);
 const owner=names.find(name=>name.endsWith('.json'))!,temporary=names.find(name=>name.endsWith('.tmp'))!;
 expect(JSON.parse(await readFile(join(lock,owner),'utf8')).released).toBe(false);
 expect(JSON.parse(await readFile(join(lock,temporary),'utf8')).released).toBe(true);
 await expect(store.withLock('k',()=>{throw Error('must not enter');},10,0)).rejects.toMatchObject({code:'EPERM'});
 expect((await lstat(lock)).isDirectory()).toBe(true);
 vi.mocked(rename).mockImplementation(actual.rename);
 await Promise.all([store.withLock('other-a',()=>1),store.withLock('other-b',()=>2)]);
 expect(await store.withLock('k',()=>7)).toBe(7);
 expect(callbacks).toBe(1);
});
it('records native shared-reader replacement and completes delayed release after close',async()=>{
 const{store,lock}=await fixture();let close:(()=>Promise<void>)|undefined,attempts=0,failures=0;
 vi.mocked(rename).mockImplementation(async(source,target)=>{
  if(String(source).endsWith('.release.tmp')){
   attempts++;try{return await actual.rename(source,target);}catch(error){failures++;throw error;}
  }
  return actual.rename(source,target);
 });
 let failure:unknown;
 try{
  try{expect(await store.withLock('k',async()=>{
   const [file]=await readdir(lock),reader=await open(join(lock,file!),'r');close=()=>reader.close();return 7;
  })).toBe(7);}catch(error){failure=error;}
  if(process.platform==='win32'){
   expect(failure).toMatchObject({code:expect.stringMatching(/EPERM|EACCES|EBUSY/)});
   expect(failures).toBe(attempts);expect(failures).toBeGreaterThan(0);
  }else{expect(failure).toBeUndefined();expect(attempts).toBe(1);expect(failures).toBe(0);}
  console.info(JSON.stringify({native_reader:'libuv-share-delete',platform:process.platform,attempts,failures}));
 }finally{await close?.();}
 expect(await store.withLock('other-key',()=>9)).toBe(9);
 await expect(lstat(lock)).rejects.toMatchObject({code:'ENOENT'});
});
it('retains successful-unlink authority across an observable directory cleanup failure',async()=>{
 const{store,lock}=await fixture();
 vi.mocked(rmdir).mockImplementation(async path=>{
  if(String(path)===lock)throw Object.assign(Error('owned cleanup denial'),{code:'EPERM'});
  return actual.rmdir(path);
 });
 await expect(store.withLock('k',()=>7)).rejects.toMatchObject({code:'EPERM'});
 expect(await readdir(lock)).toEqual([]);
 vi.mocked(rmdir).mockImplementation(actual.rmdir);
 expect(await store.withLock('other-key',()=>9)).toBe(9);
 await expect(lstat(lock)).rejects.toMatchObject({code:'ENOENT'});
});
