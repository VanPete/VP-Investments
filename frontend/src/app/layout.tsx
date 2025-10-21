import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { QueryProvider } from "@/lib/providers";
import { Toaster } from "@/components/ui/sonner";
import { Navigation } from "@/components/Navigation";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "VanPIQ Investments - Signal Rankings",
  description: "Quantitative investment signals powered by 158 factors across 6 signal groups",
  icons: {
    icon: '/favicon.svg',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} font-sans antialiased bg-gray-50`}>
        <QueryProvider>
          <Navigation />
          {children}
          <Toaster />
        </QueryProvider>
      </body>
    </html>
  );
}
