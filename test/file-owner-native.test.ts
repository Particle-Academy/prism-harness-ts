import {createHash,randomBytes} from 'node:crypto';
import {spawn} from 'node:child_process';
import {mkdtemp,lstat,mkdir,readFile,readdir,rmdir,unlink,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createInterface} from 'node:readline';
import {expect,it,vi} from 'vitest';
import {FileSessionStore} from '../src/index.js';
import * as identities from '../src/stores/file-lock-identity.js';

it.each([false,true])('executes native independent live/dead and three foreign/unknown production rows (blocked controls: %s)',async(blockedControls)=>{
 const before=identities.fileLockIdentityStatus(),started=performance.now();
 const local=await identities.nativeLockIdentity();expect(local?.pid_namespace).toBeTypeOf('string');
 console.info(JSON.stringify({native_identity_setup:true,platform:process.platform,elapsed_ms:performance.now()-started,before,status:identities.fileLockIdentityStatus()}));
 const root=await mkdtemp(join(tmpdir(),'prism-production-native-')),nonce=randomBytes(16).toString('hex');
 const spec=join(root,'spec.json');await writeFile(spec,JSON.stringify({root,nonce,package:resolve('.')}));
 if(blockedControls)for(const name of ['ready.tmp','ready.json','crash.tmp','crash.json'])await mkdir(join(root,name));
 const store=new FileSessionStore(join(root,'store')),canonical=join(root,'store',createHash('sha256').update('native-key').digest('hex').slice(0,32)+'.json.lock');
 const child=spawn(process.execPath,[resolve('test/fixtures/file-owner-native-holder.mjs'),spec],{windowsHide:true,stdio:['pipe','pipe','pipe'],timeout:30000});
 let stderr='';child.stderr.on('data',data=>{stderr+=data;});
 const exited=new Promise<number|null>((done,reject)=>{child.once('error',reject);child.once('close',done);});
 let stopped=false;
 const stop=async()=>{if(stopped||child.exitCode!==null)return;child.stdin.end(JSON.stringify({nonce,action:'crash'})+'\n');stopped=true;};
 let probes=0;
 const nativeKill=process.kill.bind(process);
 const probe=vi.spyOn(process,'kill').mockImplementation((pid,signal)=>{probes++;return nativeKill(pid,signal);});
 try{
  const lines=createInterface({input:child.stdout});
  const ready:{pid:number;nonce:string;identity:identities.LockIdentity}=await new Promise((done,reject)=>{
   const timer=setTimeout(()=>{lines.close();reject(Error('Owned holder setup failed: '+stderr));},20000);
   lines.once('line',raw=>{clearTimeout(timer);try{done(JSON.parse(raw));}catch(error){reject(error);}finally{lines.close();}});
   lines.once('close',()=>{clearTimeout(timer);reject(Error('Owned holder exited before ready: '+stderr));});
   child.stdout.once('error',reject);
  });
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
   const identityFixture=id==='both-unknown'?vi.spyOn(identities,'nativeLockIdentity').mockResolvedValue({...local!,pid_namespace:null}):undefined;
   try{await expect(store.withLock('native-key',()=>{throw Error('foreign callback');},10,0)).rejects.toMatchObject({code:'session_locked'});}
   finally{identityFixture?.mockRestore();}
   expect(probes).toBe(0);expect(await readFile(join(canonical,name))).toEqual(bytes);
   held.push({id,result:'held-unknown',probes:0,directory_unchanged:true});
   await unlink(join(canonical,name));await rmdir(canonical);
  }
  const proof={backend:'ts',production:true,identity:local,holder_identity:ready.identity,holder_pid:ready.pid,prober_pid:process.pid,live,dead,held,container_path:'unmeasured; no container created'};
  if(process.env.PRISM_FILE_OWNER_NATIVE_REPORT&&!blockedControls)await writeFile(process.env.PRISM_FILE_OWNER_NATIVE_REPORT,JSON.stringify(proof),{flag:'wx'});
  console.info(JSON.stringify(proof));
 }finally{
  probe.mockRestore();
  await stop();
  await exited;
 }
},30000);

it.each(['valid','eof','malformed','foreign','wrong-action'])('refuses invalid native pipe commands and ignores denied control reads: %s',async(command)=>{
 const root=await mkdtemp(join(tmpdir(),'prism-native-pipe-')),nonce=randomBytes(16).toString('hex');
 const spec=join(root,'spec.json');await writeFile(spec,JSON.stringify({root,nonce,package:resolve('.')}));
 const script=resolve('test/fixtures/file-owner-native-holder.mjs');
 // Inject the observed OS denial only on old IPC reads, leaving owner I/O native.
 const wrapper=`import fs from 'node:fs';import {basename} from 'node:path';import {pathToFileURL} from 'node:url';
 const original=fs.readFileSync;fs.readFileSync=function(path,...args){if(['ready.json','crash.json'].includes(basename(String(path))))throw Object.assign(Error('Denied control read'),{code:'EACCES'});return original.call(this,path,...args);};
 const [script,spec]=process.argv.slice(1);process.argv=[process.execPath,script,spec];await import(pathToFileURL(script));`;
 const child=spawn(process.execPath,['--input-type=module','-e',wrapper,script,spec],{windowsHide:true,stdio:['pipe','pipe','pipe'],timeout:30000});
 let stdout='',stderr='';child.stdout.on('data',data=>{stdout+=data;});child.stderr.on('data',data=>{stderr+=data;});
 const exited=new Promise<number|null>((done,reject)=>{child.once('error',reject);child.once('close',done);});
 const wires:Record<string,string>={valid:JSON.stringify({nonce,action:'crash'})+'\n',eof:'',malformed:'{}\n',foreign:JSON.stringify({nonce:'wrong',action:'crash'})+'\n','wrong-action':JSON.stringify({nonce,action:'release'})+'\n'};
 child.stdin.end(wires[command]);const code=await exited;
 if(command==='valid')expect(code,stderr).toBe(0);else expect(code,'Invalid input must not execute the requested crash').not.toBe(0);
 const ready=JSON.parse(stdout);expect(ready.nonce).toBe(nonce);expect(ready.pid).toBe(child.pid);expect(ready.identity).toEqual(await identities.nativeLockIdentity());
 const canonical=join(root,'store',createHash('sha256').update('native-key').digest('hex').slice(0,32)+'.json.lock');
 if(command==='valid'){const[name]=await readdir(canonical),owner=JSON.parse(await readFile(join(canonical,name!),'utf8'));expect(owner.pid).toBe(ready.pid);expect(owner.released).toBe(false);}
 else{expect(stderr).toMatch(/Wrong crash barrier|SyntaxError|command ended/);await expect(lstat(canonical)).rejects.toMatchObject({code:'ENOENT'});}
},30000);
