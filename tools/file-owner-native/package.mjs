import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const manifest = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
assert.equal(manifest.length, 1, 'one package manifest required');
const root = process.cwd(), pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
assert.equal(manifest[0].name, pkg.name);
assert.equal(manifest[0].version, pkg.version);
assert.deepEqual(Object.keys(pkg.scripts).sort(), ['build', 'test', 'typecheck'], 'no install/prepare compilation hooks');
assert.equal(pkg.scripts.build, 'tsc -p tsconfig.json');
assert.equal(manifest[0].files.length, manifest[0].entryCount);
let scanned = 0;
for (const entry of manifest[0].files) {
  assert(!/\.(?:cs|dll|node|exe|so|dylib)$|^tools\/file-owner-native\//i.test(entry.path), 'native code and test tooling must not ship');
  if (!entry.path.endsWith('.js')) continue;
  const absolute = path.resolve(root, entry.path);
  assert(absolute.startsWith(root + path.sep), 'manifest path stays within package');
  const source = fs.readFileSync(absolute, 'utf8');
  assert(!/Add-Type|DllImport|CSharpCodeProvider|Reflection\.Emit|node-gyp|prebuild-install|\bcsc\.exe\b/i.test(source),
    'published JavaScript must not embed runtime compilation: ' + entry.path);
  scanned++;
}
assert(scanned > 0, 'must inspect built production JavaScript');
assert(manifest[0].files.some(entry => entry.path === 'dist/stores/file-lock-windows-identity.js'));
console.log(JSON.stringify({ status: 'PASS', files: manifest[0].entryCount, productionJavaScriptScanned: scanned, csharpFiles: 0, compilationHooks: 0 }));
