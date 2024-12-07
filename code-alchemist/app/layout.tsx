import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// const GeistMonoVF = localFont({
//   src: "../../src/fonts/GeistMonoVF.woff",

//   variable: "--font-geist-sans",
//   weight: "100 900",
// });
// const geistMono = localFont({
//   src: "../../src/fonts/GeistVF.woff",
//   variable: "--font-geist-mono",
//   weight: "100 900",
// });

export const metadata: Metadata = {
  title: "CodeAlchemist",
  description: "Transforming logic into deployable code",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background">{children}</body>
    </html>
  );
}
