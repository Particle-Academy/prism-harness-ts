import {open} from 'node:fs/promises';
import {once} from 'node:events';
const reader=await open(process.argv[2],'r');
try{
  await reader.readFile();
  console.log('ready');
  process.stdin.resume();
  await once(process.stdin,'end');
}finally{await reader.close();}
