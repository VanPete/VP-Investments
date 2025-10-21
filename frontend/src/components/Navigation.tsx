'use client';

import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

export function Navigation() {
  const pathname = usePathname();

  const navItems = [
    { label: 'Dashboard', href: '/' },
    { label: 'Methodology', href: '/methodology' },
  ];

  return (
    <nav className="border-b border-gray-200 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo/Title */}
          <Link href="/" className="flex items-center gap-3">
            <Image 
              src="/vanpiq-logo.svg" 
              alt="VanPIQ" 
              width={120} 
              height={40}
              className="h-8 w-auto"
            />
            <span className="text-sm font-medium text-gray-600 hidden sm:inline">
              INVESTMENTS
            </span>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center space-x-6">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'text-sm font-medium transition-colors hover:text-blue-900',
                    isActive
                      ? 'text-blue-900 border-b-2 border-blue-900 pb-0.5'
                      : 'text-gray-600'
                  )}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
