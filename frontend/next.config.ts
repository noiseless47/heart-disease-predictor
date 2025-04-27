import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Config options */
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:10000/:path*',
      },
    ];
  },
};

export default nextConfig;
