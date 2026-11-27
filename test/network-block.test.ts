import { get as httpsGet, request as httpsRequest } from 'node:https';
import { request as httpRequest } from 'node:http';
import { connect, createConnection } from 'node:net';
import { connect as tlsConnect } from 'node:tls';
import { describe, expect, it } from 'vitest';

// The suite-wide network block (test/network-block-setup.ts) trips on a
// deliberate real request, through every client a test could use.

describe('the suite runs offline', () => {
  it('fetch to a real host fails', async () => {
    await expect(fetch('https://example.com/')).rejects.toThrow('A test tried to use the network (fetch)');
  });

  it('http(s).request and https.get to a real host fail', () => {
    expect(() => httpsRequest('https://example.com/')).toThrow('A test tried to use the network (https.request)');
    expect(() => httpsGet('https://example.com/')).toThrow('A test tried to use the network (https.get)');
    expect(() => httpRequest('http://example.com/')).toThrow('A test tried to use the network (http.request)');
  });

  it('net.connect, net.createConnection and tls.connect to a host and port fail', () => {
    expect(() => connect({ host: 'example.com', port: 443 })).toThrow('A test tried to use the network (net.connect)');
    expect(() => createConnection(443, 'example.com')).toThrow('A test tried to use the network (net.createConnection)');
    expect(() => tlsConnect({ host: 'example.com', port: 443 })).toThrow('A test tried to use the network (tls.connect)');
  });
});
