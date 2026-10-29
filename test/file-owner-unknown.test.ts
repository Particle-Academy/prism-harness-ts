import {mkdtemp,readFile,readdir,lstat,mkdir,writeFile,rename} from 'node:fs/promises';
import {createHash,randomBytes} from 'node:crypto';
import {spawn} from 'node:child_process';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {afterEach,expect,it,vi} from 'vitest';
import {FileSessionStore} from '../src/index.js';
import * as identity from '../src/stores/file-lock-identity.js';

afterEach(()=>vi.restoreAllMocks());
it.each(['release','crash'] as const)('a real unknown worker excludes contenders and preserves %s behavior',async action=>{
 const known=identity.nativeLockIdentity();expect(known?.pid_namespace).toBeTypeOf('string');
 const root=await mkdtemp(join(tmpdir(),'prism-unknown-worker-')),nonce=randomBytes(16).toString('hex');
 const spec=join(root,'spec.json');await writeFile(spec,JSON.stringify({root,nonce,action,package:resolve('.')}));
 const child=spawn(process.execPath,[resolve('test/fixtures/file-owner-unknown-holder.mjs'),spec],{windowsHide:true,stdio:['ignore','pipe','pipe']});
 let stderr='';child.stderr.on('data',data=>{stderr+=String(data);});
 const exited=new Promise<number|null>((done,reject)=>{child.once('error',reject);child.once('close',done);});
 const canonical=join(root,'store',createHash('sha256').update('unknown-key').digest('hex').slice(0,32)+'.json.lock');
 const send=async()=>{await writeFile(join(root,'command.tmp'),JSON.stringify({nonce,action}));await rename(join(root,'command.tmp'),join(root,'command.json'));};
 let commanded=false;
 try{
  const deadline=Date.now()+20000;let ready:{nonce:string;pid:number;injected:number}|undefined;
  while(!ready){try{ready=JSON.parse(await readFile(join(root,'ready.json'),'utf8'));}catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}if(Date.now()>deadline)throw Error('Owned unknown worker setup failed: '+stderr);if(!ready)await new Promise(done=>setTimeout(done,10));}
  expect(ready).toEqual({nonce,pid:child.pid,injected:1});
  const names=await readdir(canonical);expect(names).toHaveLength(1);
  const ownerPath=join(canonical,names[0]!),before=await readFile(ownerPath),owner=JSON.parse(before.toString('utf8'));
  expect(owner).toMatchObject({pid:child.pid,released:false,identity:{platform:process.platform,boot_id:'unknown',pid_namespace:null}});
  const probe=vi.spyOn(identity,'lockPidIsDead').mockImplementation(()=>{throw Error('Unknown owner must not authorize a PID probe');});
  const store=new FileSessionStore(join(root,'store'));let calls=0;
  await expect(store.withLock('unknown-key',()=>{calls++;},10,0)).rejects.toMatchObject({code:'session_locked'});
  expect(child.exitCode).toBeNull();expect(await readFile(ownerPath)).toEqual(before);expect(calls).toBe(0);
  await send();commanded=true;expect(await exited).toBe(0);
  if(action==='crash'){
   await new Promise(done=>setTimeout(done,1100));
   expect(owner.expires_at).toBeLessThanOrEqual(Math.floor(Date.now()/1000));
   await expect(store.withLock('unknown-key',()=>{calls++;},10,0)).rejects.toMatchObject({code:'session_locked'});
   expect(await readFile(ownerPath)).toEqual(before);expect(calls).toBe(0);
  }else{
   await expect(lstat(canonical)).rejects.toMatchObject({code:'ENOENT'});
   expect(await store.withLock('unknown-key',()=>{calls++;return 7;},10,0)).toBe(7);expect(calls).toBe(1);
  }
  expect(probe).not.toHaveBeenCalled();
 }finally{
  if(!commanded&&child.exitCode===null)await send();
  await exited;
 }
},30000);

async function fixture(){
 const root=await mkdtemp(join(tmpdir(),'prism-unknown-owner-')),store=new FileSessionStore(root);
 return {root,store,lock:join(root,createHash('sha256').update('k').digest('hex').slice(0,32)+'.json.lock')};
}
it('acquires and releases a free lock with unknown identity while excluding another store',async()=>{
 const{root,store,lock}=await fixture();
 vi.spyOn(identity,'nativeLockIdentity').mockReturnValue(null);
 const probe=vi.spyOn(identity,'lockPidIsDead').mockImplementation(()=>{throw Error('Unknown identity cannot probe a PID');});
 const warning=vi.spyOn(process,'emitWarning').mockImplementation(()=>{});
 let calls=0;
 await store.withLock('k',async()=>{
  calls++;const names=await readdir(lock);expect(names).toHaveLength(1);
  const owner=JSON.parse(await readFile(join(lock,names[0]!),'utf8'));
  expect(owner).toMatchObject({pid:process.pid,released:false,identity:{platform:process.platform,boot_id:'unknown',pid_namespace:null}});
  await expect(new FileSessionStore(root).withLock('k',()=>{throw Error('Contender entered');},10,0)).rejects.toMatchObject({code:'session_locked'});
 });
 expect(calls).toBe(1);expect(probe).not.toHaveBeenCalled();expect(warning).toHaveBeenCalled();
 await expect(lstat(lock)).rejects.toMatchObject({code:'ENOENT'});
 expect(await new FileSessionStore(root).withLock('k',()=>7,10,0)).toBe(7);
 for(const call of warning.mock.calls){expect(call[1]).toMatchObject({code:'file_lock_identity_unavailable'});expect(Object.keys(JSON.parse(String(call[0]))).sort()).toEqual(['key_digest','nonce']);}
});

it.each([false,true])('unknown owner stays held even with a released marker=%j',async released=>{
 const{store,lock}=await fixture(),nonce=randomBytes(16).toString('hex'),name='owner-'+nonce+'.json';
 const owner={version:2,pid:process.pid,identity:{platform:process.platform,boot_id:'unknown',pid_namespace:null},nonce,released,expires_at:0};
 await mkdir(lock);await writeFile(join(lock,name),JSON.stringify(owner)+'\n');
 const before=await readFile(join(lock,name));
 const probe=vi.spyOn(identity,'lockPidIsDead').mockImplementation(()=>{throw Error('Unknown owner cannot authorize a probe');});
 await expect(store.withLock('k',()=>{throw Error('Unknown owner was reclaimed');},10,0)).rejects.toMatchObject({code:'session_locked'});
 expect(probe).not.toHaveBeenCalled();expect(await readFile(join(lock,name))).toEqual(before);
});

it('an unknown prober never reclaims a known released owner',async()=>{
 const{store,lock}=await fixture(),known=identity.nativeLockIdentity();expect(known?.pid_namespace).toBeTypeOf('string');
 const nonce=randomBytes(16).toString('hex'),name='owner-'+nonce+'.json';
 const owner={version:2,pid:process.pid,identity:known,nonce,released:true,expires_at:0};
 await mkdir(lock);await writeFile(join(lock,name),JSON.stringify(owner)+'\n');
 const before=await readFile(join(lock,name));
 vi.spyOn(identity,'nativeLockIdentity').mockReturnValue(null);
 const probe=vi.spyOn(identity,'lockPidIsDead').mockImplementation(()=>{throw Error('Unknown prober cannot consult a PID');});
 vi.spyOn(process,'emitWarning').mockImplementation(()=>{});
 await expect(store.withLock('k',()=>{throw Error('Unknown prober reclaimed');},10,0)).rejects.toMatchObject({code:'session_locked'});
 expect(probe).not.toHaveBeenCalled();expect(await readFile(join(lock,name))).toEqual(before);
});
