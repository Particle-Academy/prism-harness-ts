import {beforeAll,expect} from 'vitest';

// Metadata can legitimately consume its 15s bound on a cold Windows host.
// Keep that setup outside individual 5s test bodies; query-contract unit tests own their mocks.
beforeAll(async()=>{
 if(process.platform!=='win32')return;
 const path=expect.getState().testPath??'';
 if(/file-owner-identity-(cache|boundary)\.test\.ts$/.test(path))return;
 const {fileLockIdentityStatus,nativeLockIdentity}=await import('../src/stores/file-lock-identity.js');
 const start=performance.now();nativeLockIdentity();
 console.info(JSON.stringify({native_identity_setup:true,platform:process.platform,
   elapsed_ms:performance.now()-start,status:fileLockIdentityStatus()}));
},20000);
