/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "#fff8ef",
        ink: "#16110a",
        accent: "#dd6b20",
        accentSoft: "#ffd7b8",
        success: "#1f7a52",
        danger: "#b83232",
      },
      boxShadow: {
        panel: "0 24px 80px rgba(31, 22, 12, 0.12)",
      },
    },
  },
  plugins: [],
};
