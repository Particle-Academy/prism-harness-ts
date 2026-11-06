import {defineConfig} from 'vitest/config';

export default defineConfig({test:{setupFiles:['test/native-identity-setup.ts'],testTimeout:process.platform==='win32'?15000:5000}});
