/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "Avenir", "Helvetica", "Arial", "sans-serif"],
      },
      colors: {
        surface: "#faf8f5",
        ink: "#1a1510",
        accent: "#e07826",
        "accent-hover": "#c96a1f",
        "accent-soft": "#fff3e8",
        success: "#16794a",
        "success-soft": "#ecfdf3",
        danger: "#c53030",
        "danger-soft": "#fef2f2",
        muted: "#78716c",
        border: "#e8e5e1",
      },
      boxShadow: {
        card: "0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.06)",
        "card-hover": "0 2px 8px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.08)",
      },
      maxWidth: {
        page: "1280px",
      },
      borderRadius: {
        card: "24px",
        control: "16px",
        pill: "9999px",
      },
      spacing: {
        4.5: "1.125rem", /* 18px */
      },
    },
  },
  plugins: [],
};
