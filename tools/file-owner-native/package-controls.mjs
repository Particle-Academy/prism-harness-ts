import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import test from 'node:test';

const checker=fileURLToPath(new URL('./package.mjs',import.meta.url));
function fixture(alter=()=>{}){
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'prism-package-controls-'));
 const pkg={name:'@particle-academy/prism-harness',version:'0.8.0',scripts:{build:'tsc -p tsconfig.json',test:'vitest run',typecheck:'tsc -p tsconfig.typecheck.json'}};
 const files=[{path:'dist/index.js'},{path:'dist/stores/file-lock-windows-identity.js'}],manifest=[{name:pkg.name,version:pkg.version,files,entryCount:2}];
 fs.mkdirSync(path.join(root,'dist/stores'),{recursive:true});
 fs.writeFileSync(path.join(root,'dist/index.js'),'export const fixture = true;');
 fs.writeFileSync(path.join(root,files[1].path),'export const fixedMetadataOnly = true;');
 alter({root,pkg,manifest,files});
 fs.writeFileSync(path.join(root,'package.json'),JSON.stringify(pkg));
 fs.writeFileSync(path.join(root,'manifest.json'),JSON.stringify(manifest));
 return spawnSync(process.execPath,[checker,path.join(root,'manifest.json')],{cwd:root,encoding:'utf8',timeout:10000,windowsHide:true});
}
test('accepts inspected runtime JavaScript with no native test files or compilation hook',()=>assert.equal(fixture().status,0));
for(const[name,alter]of[
 ['embedded compiler',({root,files})=>fs.writeFileSync(path.join(root,files[1].path),'const command = "Add-Type";')],
 ['C# in tarball',({manifest,files})=>{files.push({path:'tools/file-owner-native/windows-native.cs'});manifest[0].entryCount++;}],
 ['native addon in tarball',({manifest,files})=>{files.push({path:'dist/native.node'});manifest[0].entryCount++;}],
 ['lifecycle compile hook',({pkg})=>pkg.scripts.prepare='csc.exe unsafe.cs'],
 ['missing built runtime',({manifest})=>{manifest[0].files=[];manifest[0].entryCount=0;}],
 ['escaping manifest path',({manifest,files})=>{files.push({path:'../outside.js'});manifest[0].entryCount++;}],
])test('refuses '+name,()=>assert.notEqual(fixture(alter).status,0));
