import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === 'production';
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || (isProd
  ? 'https://heart-disease-predictor-iizp.onrender.com'
  : 'http://localhost:10000');

const nextConfig: NextConfig = {
  /* Config options */
  async rewrites() {
    return [
      {
        source: '/predict',
        destination: `${backendUrl}/predict`,
      },
      {
        source: '/api/:path*',
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
