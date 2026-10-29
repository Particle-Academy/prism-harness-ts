// Test-only metadata fault injection; production lock and FileSessionStore execute unchanged.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {registerHooks} from 'node:module';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
const spec=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
assert.equal(typeof spec.nonce,'string');assert.match(spec.nonce,/^[0-9a-f]{32}$/);
assert.ok(['release','crash'].includes(spec.action));
const target=pathToFileURL(join(spec.package,'dist/stores/file-lock-identity.js')).href;
let injected=0;
registerHooks({load(url,context,nextLoad){
 if(url!==target)return nextLoad(url,context);
 injected++;
 return {format:'module',shortCircuit:true,source:`
 export function nativeLockIdentity(){return null;}
 export function sameLockIdentity(){return false;}
 export function lockPidIsDead(){throw Error('Unknown identity must not probe a PID');}
 export function fileLockIdentityStatus(){return {sampled:true,recovery_available:false,definitive:false,retry_after_ms:60000,sample_duration_ms:0,code:'file_lock_identity_unavailable'};}
 `};
}});
const {FileSessionStore}=await import(pathToFileURL(join(spec.package,'dist/index.js')));
const store=new FileSessionStore(join(spec.root,'store'));
await store.withLock('unknown-key',async()=>{
 assert.equal(injected,1);
 fs.writeFileSync(join(spec.root,'ready.tmp'),JSON.stringify({nonce:spec.nonce,pid:process.pid,injected}),{flag:'wx'});
 fs.renameSync(join(spec.root,'ready.tmp'),join(spec.root,'ready.json'));
 const deadline=Date.now()+25000;
 for(;;){
  try{
   const command=JSON.parse(fs.readFileSync(join(spec.root,'command.json'),'utf8'));
   assert.deepEqual(command,{nonce:spec.nonce,action:spec.action});
   if(command.action==='crash')process.exit(0);
   return;
  }catch(error){if(error.code!=='ENOENT')throw error;}
  if(Date.now()>deadline)throw Error('Owned unknown holder barrier timed out');
  await new Promise(done=>setTimeout(done,10));
 }
},1);
