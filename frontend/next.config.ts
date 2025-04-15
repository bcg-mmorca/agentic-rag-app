import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  /* config options here */
  output: 'standalone',
  experimental: {
    serverActions: {
      bodySizeLimit: `${Number(process.env.BODY_SIZE_LIMIT || '10')}MB`,
    },
  },
};

export default nextConfig;
