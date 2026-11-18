/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#faf2f9',
          100: '#f1e0ef',
          200: '#e6c8e2',
          300: '#d5a4cc',
          400: '#b773ad',
          500: '#9b4c91',
          600: '#803874',
          700: '#6c2c62',
          800: '#56214f',
          900: '#441b44',
        },
        sentiment: {
          positif: '#22c55e',
          netral: '#3b82f6',
          negatif: '#ef4444',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
