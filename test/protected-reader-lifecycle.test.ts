import {randomBytes} from 'node:crypto';
import {spawn} from 'node:child_process';
import {expect,it} from 'vitest';
import {observeProtectedReader,type ReaderLifecycle} from './support/protected-reader.js';

function child(code:string,nonce:string){
  return spawn(process.execPath,['--input-type=module','-e',code,nonce],{windowsHide:true,stdio:['pipe','pipe','pipe']});
}
function lifecycle(error:unknown):ReaderLifecycle {
  expect(error).toBeInstanceOf(Error);
  const value=(error as Error&{readerLifecycle:ReaderLifecycle}).readerLifecycle;
  expect(Object.keys(value).sort()).toEqual(['exitCode','ioError','phase','phaseAtExit','ready','reason','signal','spawnError','stderrTail','timedOutPhase'].sort());
  expect((error as Error).message).toContain(JSON.stringify(value));return value;
}

it('gives startup its own budget before starting the hold window',async()=>{
  const nonce=randomBytes(16).toString('hex');
  const worker=child("import{createInterface}from'node:readline';const input=createInterface({input:process.stdin});setTimeout(()=>{console.log(JSON.stringify({status:'ready',nonce:process.argv[1],pid:process.pid}));input.once('line',raw=>{const v=JSON.parse(raw);if(v.nonce!==process.argv[1]||v.action!=='close')process.exit(9);input.close();});},2500);",nonce);
  const close=await observeProtectedReader(worker,nonce,{startup:20000,hold:2000});await close();
},30000);

it('reports the real exit code and a bounded stderr tail before readiness',async()=>{
  const nonce=randomBytes(16).toString('hex'),worker=child("process.stderr.write('x'.repeat(4096)+'actual-reader-error',()=>process.exit(7))",nonce);
  try{await observeProtectedReader(worker,nonce,{startup:20000,hold:10000});expect.fail('Exited reader accepted');}
  catch(error){const value=lifecycle(error);expect(value.exitCode).toBe(7);expect(value.signal).toBeNull();expect(value.spawnError).toBeNull();expect(value.phaseAtExit).toBe('startup');expect(value.stderrTail.endsWith('actual-reader-error')).toBe(true);expect(value.stderrTail.length).toBeLessThanOrEqual(2048);expect(value.timedOutPhase).toBeNull();}
},30000);

it('reports spawn errors instead of an unexplained null exit',async()=>{
  const worker=spawn('prism-g244-nonexistent-'+randomBytes(16).toString('hex'),[],{windowsHide:true,stdio:['pipe','pipe','pipe']});
  try{await observeProtectedReader(worker,'unused',{startup:20000,hold:10000});expect.fail('Missing executable accepted');}
  catch(error){const value=lifecycle(error);expect(value.spawnError?.code).toBe('ENOENT');expect(value.ready).toBe(false);expect(value.signal).toBeNull();expect(value.stderrTail).toBe('');}
},30000);

it.each(['startup','hold'] as const)('records the %s deadline and actual termination details',async(phase)=>{
  const nonce=randomBytes(16).toString('hex');
  const code=phase==='startup'?"setInterval(()=>{},1000)":"console.log(JSON.stringify({status:'ready',nonce:process.argv[1],pid:process.pid}));process.stdin.resume();setInterval(()=>{},1000)";
  const worker=child(code,nonce);
  try{const close=await observeProtectedReader(worker,nonce,{startup:phase==='startup'?200:20000,hold:200});await close();expect.fail('Unresponsive reader accepted');}
  catch(error){const value=lifecycle(error);expect(value.timedOutPhase).toBe(phase);expect(value.spawnError).toBeNull();expect(value.signal==='SIGKILL'||value.exitCode!==null).toBe(true);if(process.platform!=='win32')expect(value.signal).toBe('SIGKILL');expect(value.ready).toBe(phase==='hold');}
},30000);

it.each(['malformed','foreign'] as const)('refuses a %s ready barrier',async(kind)=>{
  const nonce=randomBytes(16).toString('hex'),worker=child(kind==='malformed'?"console.log('ready');setInterval(()=>{},1000)":"console.log(JSON.stringify({status:'ready',nonce:'foreign',pid:process.pid}));setInterval(()=>{},1000)",nonce);
  try{await observeProtectedReader(worker,nonce,{startup:20000,hold:10000});expect.fail('Invalid ready accepted');}
  catch(error){const value=lifecycle(error);expect(value.reason).toBe('invalid reader ready barrier');expect(value.ready).toBe(false);expect(value.spawnError).toBeNull();}
},30000);
