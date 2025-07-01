import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ["d7e0-183-82-1-156.ngrok-free.app"], // 👈 Add your Ngrok host here
    proxy: {
      '/api': 'http://localhost:5000', // proxies /api/* to FastAPI
    },
  
  },

}) 
