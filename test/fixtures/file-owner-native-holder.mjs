import fs from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
const spec=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const {FileSessionStore}=await import(pathToFileURL(spec.package+'/dist/index.js'));
const {nativeLockIdentity}=await import(pathToFileURL(spec.package+'/dist/stores/file-lock-identity.js'));
const identity=nativeLockIdentity();
if(!identity?.pid_namespace)throw Error('Native holder identity unavailable');
const store=new FileSessionStore(join(spec.root,'store'));
await store.withLock('native-key',async()=>{
 fs.writeFileSync(join(spec.root,'ready.tmp'),JSON.stringify({nonce:spec.nonce,pid:process.pid,identity}),{flag:'wx'});
 fs.renameSync(join(spec.root,'ready.tmp'),join(spec.root,'ready.json'));
 const deadline=Date.now()+25000;
 for(;;){
  try{const message=JSON.parse(fs.readFileSync(join(spec.root,'crash.json'),'utf8'));if(message.nonce!==spec.nonce)throw Error('Wrong crash barrier');process.exit(0);}
  catch(error){if(error.code!=='ENOENT')throw error;}
  if(Date.now()>deadline)throw Error('Owned holder barrier timed out');
  await new Promise(done=>setTimeout(done,10));
 }
},1);
