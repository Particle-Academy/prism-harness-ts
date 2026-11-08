import fs from 'node:fs';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createInterface} from 'node:readline';
const spec=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const {FileSessionStore}=await import(pathToFileURL(spec.package+'/dist/index.js'));
const {nativeLockIdentity}=await import(pathToFileURL(spec.package+'/dist/stores/file-lock-identity.js'));
const identity=await nativeLockIdentity();
if(!identity?.pid_namespace)throw Error('Native holder identity unavailable');
const store=new FileSessionStore(join(spec.root,'store'));
await store.withLock('native-key',async()=>{
 const lines=createInterface({input:process.stdin});
 let timer;
 const command=new Promise((done,reject)=>{
  timer=setTimeout(()=>reject(Error('Owned holder barrier timed out')),25000);
  lines.once('line',done);
  lines.once('close',()=>reject(Error('Owned holder command ended before a line')));
  process.stdin.once('error',reject);
 });
 process.stdout.write(JSON.stringify({nonce:spec.nonce,pid:process.pid,identity})+'\n');
 try{
  const message=JSON.parse(await command);
  if(message?.nonce!==spec.nonce||message?.action!=='crash'||Object.keys(message).sort().join(',')!=='action,nonce')throw Error('Wrong crash barrier');
  process.exit(0);
 }finally{clearTimeout(timer);lines.close();}
},1);
