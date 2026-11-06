import {randomBytes} from 'node:crypto';
import {spawn, type ChildProcessWithoutNullStreams} from 'node:child_process';

type Phase = 'startup' | 'holding' | 'closing' | 'closed';
type Failure = {name:string; code:string|null; message:string};
export type ReaderLifecycle = {
  phase:Phase; phaseAtExit:Phase|null; ready:boolean; exitCode:number|null;
  signal:NodeJS.Signals|null; spawnError:Failure|null; ioError:Failure|null;
  stderrTail:string; timedOutPhase:'startup'|'hold'|null; reason:string|null;
};
type Budgets = {startup:number; hold:number};

function describe(error:Error):Failure {
  return {name:error.name, code:(error as NodeJS.ErrnoException).code??null, message:error.message};
}
function failure(lifecycle:ReaderLifecycle):Error {
  return Object.assign(Error('Owned reader failed: '+JSON.stringify(lifecycle)), {readerLifecycle:lifecycle});
}

/** Observe an owned pipe protocol; a protocol test alone is not sharing-denial proof. */
export async function observeProtectedReader(
  child:ChildProcessWithoutNullStreams, nonce:string, budgets:Budgets,
):Promise<()=>Promise<void>> {
  const lifecycle:ReaderLifecycle={phase:'startup',phaseAtExit:null,ready:false,exitCode:null,
    signal:null,spawnError:null,ioError:null,stderrTail:'',timedOutPhase:null,reason:null};
  let stdout='', timer:ReturnType<typeof setTimeout>|undefined;
  let readyDone!:(ready:boolean)=>void, exitDone!:(value:ReaderLifecycle)=>void;
  const ready=new Promise<boolean>(done=>{readyDone=done;});
  const exited=new Promise<ReaderLifecycle>(done=>{exitDone=done;});
  const stop=(reason:string)=>{
    lifecycle.reason??=reason;
    if(child.pid!==undefined&&child.exitCode===null&&child.signalCode===null)child.kill('SIGKILL');
  };
  const arm=(phase:'startup'|'hold',milliseconds:number)=>{
    clearTimeout(timer);
    timer=setTimeout(()=>{lifecycle.timedOutPhase=phase;stop('reader '+phase+' deadline exceeded');},milliseconds);
  };
  child.stderr.on('data',data=>{lifecycle.stderrTail=(lifecycle.stderrTail+String(data)).slice(-2048);});
  child.on('error',error=>{lifecycle.spawnError=describe(error);stop('reader spawn error');});
  child.stdin.on('error',error=>{lifecycle.ioError=describe(error);stop('reader input error');});
  child.stdout.on('error',error=>{lifecycle.ioError=describe(error);stop('reader output error');});
  child.stdout.on('data',data=>{
    if(lifecycle.reason)return;
    if(lifecycle.ready){if(String(data).trim())stop('unexpected reader output after ready');return;}
    stdout+=String(data);
    if(Buffer.byteLength(stdout)>4096){stop('reader readiness output exceeded bound');return;}
    const end=stdout.indexOf('\n');if(end<0)return;
    try {
      const value:unknown=JSON.parse(stdout.slice(0,end));
      if(typeof value!=='object'||value===null)throw Error('invalid ready');
      const row=value as Record<string,unknown>;
      if(Object.keys(row).sort().join(',')!=='nonce,pid,status'||row.status!=='ready'||row.nonce!==nonce||row.pid!==child.pid||stdout.slice(end+1).trim())throw Error('invalid ready');
    } catch {stop('invalid reader ready barrier');return;}
    lifecycle.ready=true;lifecycle.phase='holding';arm('hold',budgets.hold);readyDone(true);
  });
  child.once('close',(code,signal)=>{
    clearTimeout(timer);lifecycle.exitCode=code;lifecycle.signal=signal;lifecycle.phaseAtExit=lifecycle.phase;
    if(!lifecycle.ready)lifecycle.reason??='reader exited before ready';
    else if(lifecycle.phase!=='closing')lifecycle.reason??='reader exited before close barrier';
    if(code!==0||signal!==null)lifecycle.reason??='reader exited unsuccessfully';
    lifecycle.phase='closed';readyDone(false);exitDone(structuredClone(lifecycle));
  });
  arm('startup',budgets.startup);
  if(!await ready)throw failure(await exited);
  return async()=>{
    if(lifecycle.phase==='holding'&&child.exitCode===null&&child.signalCode===null){
      lifecycle.phase='closing';child.stdin.end(JSON.stringify({nonce,action:'close'})+'\n');
    }
    const result=await exited;
    if(result.reason!==null||result.spawnError!==null||result.ioError!==null||result.timedOutPhase!==null||result.exitCode!==0||result.signal!==null)throw failure(result);
  };
}

/** FileShare.ReadWrite deliberately omits Delete: keep the actual .NET open. */
export async function protectedWindowsReader(path:string):Promise<()=>Promise<void>> {
  const nonce=randomBytes(16).toString('hex');
  const command="$ErrorActionPreference='Stop'; $data=[Console]::In.ReadLine() | ConvertFrom-Json; $stream=[IO.File]::Open($data.path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite); try { [Console]::Out.WriteLine((@{status='ready';nonce=$data.nonce;pid=$PID} | ConvertTo-Json -Compress)); [Console]::Out.Flush(); $close=[Console]::In.ReadLine() | ConvertFrom-Json; if ($close.nonce -ne $data.nonce -or $close.action -ne 'close') { throw 'Wrong close barrier' } } finally { $stream.Dispose() }";
  const child=spawn('powershell.exe',['-NoLogo','-NoProfile','-NonInteractive','-Command',command],{windowsHide:true,stdio:['pipe','pipe','pipe']});
  const result=observeProtectedReader(child,nonce,{startup:20000,hold:10000});
  child.stdin.write(JSON.stringify({path,nonce})+'\n');
  return result;
}
