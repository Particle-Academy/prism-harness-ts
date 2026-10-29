import {mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {expect,it} from 'vitest';
import {AgentRuntime,FileSessionStore,ModeRegistry,PrismHarness,ToolRegistry} from '../src/index.js';

const cases = [
  {name:'strict legacy refuses widening',narrow:true,named:false,names:['outside'],expected:null},
  {name:'strict legacy wildcard retains the mode ceiling',narrow:true,named:false,names:['*'],expected:['inside']},
  {name:'strict legacy empty list offers nothing',narrow:true,named:false,names:[],expected:[]},
  {name:'default legacy retains caller widening',narrow:false,named:false,names:['outside'],expected:['outside']},
  {name:'default legacy retains native empty-list behavior',narrow:false,named:false,names:[],expected:[]},
  {name:'named sessions refuse widening without the option',narrow:false,named:true,names:['outside'],expected:null},
  {name:'named wildcard retains the mode ceiling',narrow:false,named:true,names:['*'],expected:['inside']},
];
for(const structured of [false,true])it.each(cases)('$name (structured='+structured+')',async test=>{
  const directory=await mkdtemp(join(tmpdir(),'prism-tool-narrowing-')),store=new FileSessionStore(directory);
  const h=new PrismHarness({drivers:{owned:()=>store},stores:{ephemeral:'owned',durable:'owned'}});
  const pending=test.named?h.forAgent('support'):h,session=pending.for({type:'test',id:'narrowing'}).session('tools');
  const tool=(name:string)=>({name,handle:()=>{throw Error('No handler may run');}});
  const tools=new ToolRegistry().register(tool('inside')).register(tool('outside'));
  tools.forAgent('support').register(tool('inside'),'support').register(tool('outside'),'support');
  const modes=new ModeRegistry({default:'chat',modes:{chat:{tools:['inside']}}});
  modes.forAgent('support').register('chat',{tools:['inside']},'support').setDefault('chat');
  const requests:string[][]=[];
  const runtime=new AgentRuntime({tools,modes,narrowRequestedTools:test.narrow,client:async request=>{
    requests.push(request.tools.map(tool=>tool.name));return{text:'done',finishReason:'stop',structured:{answer:'done'},costUsd:0};
  }});
  const turn=structured?runtime.sendStructured(session,'probe',{type:'object'},test.names):runtime.send(session,'probe',test.names);
  if(test.expected===null){await expect(turn).rejects.toMatchObject({code:'tool_not_available'});expect(requests).toEqual([]);}
  else{await turn;expect(requests).toEqual([test.expected]);}
});
