import * as React from 'react'
import { cn } from '@/lib/utils'

type BadgeVariant = 'default' | 'secondary' | 'destructive'

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  const base = 'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold'
  const variants: Record<BadgeVariant, string> = {
    default: 'border-transparent bg-gray-100 text-gray-800',
    secondary: 'border-transparent bg-blue-100 text-blue-800',
    destructive: 'border-transparent bg-red-100 text-red-800',
  }
  return <span className={cn(base, variants[variant], className)} {...props} />
}

export type { BadgeVariant }
