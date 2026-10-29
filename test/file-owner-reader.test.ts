import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdtemp, open, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { expect, it } from 'vitest';
import { FileSessionStore } from '../src/index.js';

async function blockingReader(path:string):Promise<()=>Promise<void>>{
 if(process.platform!=='win32'){const handle=await open(path,'r');return async()=>{await handle.close();};}
 const command="$ErrorActionPreference='Stop'; $data=[Console]::In.ReadLine() | ConvertFrom-Json; $stream=[IO.File]::Open($data.path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite); try { [Console]::Out.WriteLine('ready'); [Console]::Out.Flush(); if ([Console]::In.ReadLine() -ne 'close') { throw 'Missing close barrier' } } finally { $stream.Dispose() }";
 const child=spawn('powershell.exe',['-NoLogo','-NoProfile','-NonInteractive','-Command',command],{windowsHide:true,stdio:['pipe','pipe','pipe'],timeout:10000});
 let stderr='',stdout='';child.stderr.on('data',data=>{stderr+=data;});
 const exited=new Promise<void>((done,reject)=>{child.once('error',reject);child.once('close',code=>{code===0?done():reject(Error('Owned reader failed: '+code+' '+stderr));});});
 const ready=new Promise<void>((done,reject)=>{child.stdout.on('data',data=>{stdout+=data;if(stdout.includes('ready'))done();});exited.then(()=>{if(!stdout.includes('ready'))reject(Error('Reader exited before barrier'));},reject);});
 child.stdin.write(JSON.stringify({path})+'\n');await ready;
 return async()=>{child.stdin.end('close\n');await exited;};
}
it('finishes a pending release after a real reader closes while the owner stays alive',async()=>{
 const directory=await mkdtemp(join(tmpdir(),'prism-reader-release-')),store=new FileSessionStore(directory);
 const lock=join(directory,createHash('sha256').update('k').digest('hex').slice(0,32)+'.json.lock');
 const ownerPid=process.pid;let close:(()=>Promise<void>)|undefined,callbacks=0,failure:unknown;
 try{
  try{await store.withLock('k',async()=>{
   callbacks++;const files=await readdir(lock);close=await blockingReader(join(lock,files.find(name=>name.endsWith('.json'))!));
  },-1);}catch(error){failure=error;}
  expect(callbacks).toBe(1);expect(process.pid).toBe(ownerPid);
  if(process.platform==='win32')expect(failure).toMatchObject({code:expect.stringMatching(/EPERM|EACCES|EBUSY/)});
  else expect(failure).toBeUndefined();
 }finally{await close?.();}
 expect(await store.withLock('other-key',()=> 'owner-alive')).toBe('owner-alive');
 const entry=resolve('dist/index.js'),script="import{pathToFileURL}from'node:url';const{FileSessionStore}=await import(pathToFileURL(process.argv[1]).href);let callbacks=0,code=null;try{await new FileSessionStore(process.argv[2]).withLock('k',()=>{callbacks++;},10,0.1);}catch(e){code=e.code;}console.log(JSON.stringify({callbacks,code,pid:process.pid}));";
 const child=spawn(process.execPath,['--input-type=module','-e',script,entry,directory],{windowsHide:true,stdio:['ignore','pipe','pipe'],timeout:10000});
 let stdout='',stderr='';child.stdout.on('data',data=>{stdout+=data;});child.stderr.on('data',data=>{stderr+=data;});
 const code=await new Promise<number|null>((done,reject)=>{child.once('error',reject);child.once('close',done);});
 expect(code,stderr).toBe(0);
 const result=JSON.parse(stdout);expect(result).toMatchObject({callbacks:1,code:null});
 expect(result.pid).not.toBe(ownerPid);expect(process.pid).toBe(ownerPid);
},20000);
