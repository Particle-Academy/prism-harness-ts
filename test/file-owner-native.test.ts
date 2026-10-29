import {createHash,randomBytes} from 'node:crypto';
import {spawn} from 'node:child_process';
import {mkdtemp,lstat,mkdir,readFile,readdir,rename,rmdir,unlink,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {expect,it,vi} from 'vitest';
import {FileSessionStore} from '../src/index.js';
import * as identities from '../src/stores/file-lock-identity.js';

it('executes native independent live/dead and three foreign/unknown production rows',async()=>{
 const before=identities.fileLockIdentityStatus(),started=performance.now();
 const local=identities.nativeLockIdentity();expect(local?.pid_namespace).toBeTypeOf('string');
 console.info(JSON.stringify({native_identity_setup:true,platform:process.platform,elapsed_ms:performance.now()-started,before,status:identities.fileLockIdentityStatus()}));
 const root=await mkdtemp(join(tmpdir(),'prism-production-native-')),nonce=randomBytes(16).toString('hex');
 const spec=join(root,'spec.json');await writeFile(spec,JSON.stringify({root,nonce,package:resolve('.')}));
 const store=new FileSessionStore(join(root,'store')),canonical=join(root,'store',createHash('sha256').update('native-key').digest('hex').slice(0,32)+'.json.lock');
 const child=spawn(process.execPath,[resolve('test/fixtures/file-owner-native-holder.mjs'),spec],{windowsHide:true,stdio:['ignore','pipe','pipe'],timeout:30000});
 let stderr='';child.stderr.on('data',data=>{stderr+=data;});
 const exited=new Promise<number|null>((done,reject)=>{child.once('error',reject);child.once('close',done);});
 let stopped=false;
 const stop=async()=>{if(stopped)return;await writeFile(join(root,'crash.tmp'),JSON.stringify({nonce}));await rename(join(root,'crash.tmp'),join(root,'crash.json'));stopped=true;};
 let probes=0;
 const nativeKill=process.kill.bind(process);
 const probe=vi.spyOn(process,'kill').mockImplementation((pid,signal)=>{probes++;return nativeKill(pid,signal);});
 try{
  const deadline=Date.now()+20000;let ready:{pid:number;nonce:string;identity:identities.LockIdentity}|undefined;
  while(!ready){try{ready=JSON.parse(await readFile(join(root,'ready.json'),'utf8'));}catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}if(Date.now()>deadline)throw Error('Owned holder setup failed: '+stderr);if(!ready)await new Promise(done=>setTimeout(done,10));}
  expect(ready.nonce).toBe(nonce);expect(ready.pid).toBe(child.pid);expect(ready.identity).toEqual(local);
  const [oldName]=await readdir(canonical),oldBytes=await readFile(join(canonical,oldName!)),old=JSON.parse(oldBytes.toString('utf8'));
  expect(old.pid).toBe(ready.pid);expect(old.released).toBe(false);
  await new Promise(done=>setTimeout(done,1100));
  expect(old.expires_at).toBeLessThanOrEqual(Math.floor(Date.now()/1000));
  probes=0;let callbacks=0;
  await expect(store.withLock('native-key',()=>{callbacks++;},10,0)).rejects.toMatchObject({code:'session_locked'});
  expect(callbacks).toBe(0);expect(probes).toBe(1);expect(child.exitCode).toBeNull();
  expect(await readFile(join(canonical,oldName!))).toEqual(oldBytes);
  const live={result:'held',probes};
  await stop();
  expect(await exited,stderr).toBe(0);
  expect(await readFile(join(canonical,oldName!))).toEqual(oldBytes);
  probes=0;let replacementNonce='';
  await store.withLock('native-key',async()=>{
   callbacks++;const[name]=await readdir(canonical),replacement=JSON.parse(await readFile(join(canonical,name!),'utf8'));
   expect(replacement.pid).toBe(process.pid);expect(replacement.identity).toEqual(local);
   expect(replacement.nonce).not.toBe(old.nonce);replacementNonce=replacement.nonce;
  },10,0.5);
  expect(callbacks).toBe(1);expect(probes).toBe(1);
  const dead={result:'nonce-bound-reclaimed',probes,replacement_nonce:replacementNonce};
  await expect(lstat(canonical)).rejects.toMatchObject({code:'ENOENT'});
  const held=[];
  for(const id of ['foreign-realm','foreign-realm-released','both-unknown']){
   const ownerIdentity={...local!,pid_namespace:id==='both-unknown'?null:local!.pid_namespace+':foreign'};
   const fixtureNonce=randomBytes(16).toString('hex'),name='owner-'+fixtureNonce+'.json';
   await mkdir(canonical);const bytes=Buffer.from(JSON.stringify({version:2,pid:process.pid,identity:ownerIdentity,nonce:fixtureNonce,released:id!=='foreign-realm',expires_at:0})+'\n');
   await writeFile(join(canonical,name),bytes);probes=0;
   const identityFixture=id==='both-unknown'?vi.spyOn(identities,'nativeLockIdentity').mockReturnValue({...local!,pid_namespace:null}):undefined;
   try{await expect(store.withLock('native-key',()=>{throw Error('foreign callback');},10,0)).rejects.toMatchObject({code:'session_locked'});}
   finally{identityFixture?.mockRestore();}
   expect(probes).toBe(0);expect(await readFile(join(canonical,name))).toEqual(bytes);
   held.push({id,result:'held-unknown',probes:0,directory_unchanged:true});
   await unlink(join(canonical,name));await rmdir(canonical);
  }
  const proof={backend:'ts',production:true,identity:local,holder_identity:ready.identity,holder_pid:ready.pid,prober_pid:process.pid,live,dead,held,container_path:'unmeasured; no container created'};
  if(process.env.PRISM_FILE_OWNER_NATIVE_REPORT)await writeFile(process.env.PRISM_FILE_OWNER_NATIVE_REPORT,JSON.stringify(proof),{flag:'wx'});
  console.info(JSON.stringify(proof));
 }finally{
  probe.mockRestore();
  await stop();
  await exited;
 }
},30000);
