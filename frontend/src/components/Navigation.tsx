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
    <nav className="border-b border-gray-200 bg-white dark:bg-gray-950 dark:border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link 
            href="/" 
            className="flex items-center pl-3 group"
          >
            <Image 
              src="/vanpiq-logo.svg" 
              alt="VanPIQ" 
              width={100} 
              height={34}
              className="h-[34px] w-[100px] transition-all duration-300 group-hover:drop-shadow-[0_0_8px_rgba(0,174,239,0.5)]"
            />
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
                    'text-sm font-medium transition-all hover:text-blue-600 dark:hover:text-blue-400 pb-0.5',
                    isActive
                      ? 'text-blue-900 dark:text-blue-400 border-b-2 border-gradient-to-r from-[#001F3F] to-[#00AEEF]'
                      : 'text-gray-600 dark:text-gray-400'
                  )}
                  style={isActive ? {
                    borderImage: 'linear-gradient(to right, #001F3F, #00AEEF) 1',
                  } : undefined}
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
