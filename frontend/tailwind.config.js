/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        sharepoint: {
          blue: '#0078D4',
          light: '#F3F2F1',
          header: '#005A9E',
        }
      }
    },
  },
  plugins: [],
}
