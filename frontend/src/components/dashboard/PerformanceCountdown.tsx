'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Clock } from 'lucide-react';

interface PerformanceCountdownProps {
  signalCreatedAt?: string;
}

export function PerformanceCountdown({ signalCreatedAt }: PerformanceCountdownProps) {
  const [timeUntilNext, setTimeUntilNext] = useState<{
    days: number;
    hours: number;
    minutes: number;
    seconds: number;
    nextMilestone: string;
  } | null>(null);

  useEffect(() => {
    if (!signalCreatedAt) return;

    const updateCountdown = () => {
      const signalDate = new Date(signalCreatedAt);
      const now = new Date();
      const msElapsed = now.getTime() - signalDate.getTime();
      const daysElapsed = msElapsed / (1000 * 60 * 60 * 24);

      // Determine next milestone
      const milestones = [
        { name: '1D', days: 1 },
        { name: '3D', days: 3 },
        { name: '7D', days: 7 },
        { name: '10D', days: 10 },
        { name: '14D', days: 14 },
        { name: '30D', days: 30 },
        { name: '90D', days: 90 },
      ];

      const nextMilestone = milestones.find(m => daysElapsed < m.days);

      if (!nextMilestone) {
        // All milestones complete
        setTimeUntilNext(null);
        return;
      }

      // Calculate time until next milestone
      const nextMilestoneDate = new Date(signalDate.getTime() + nextMilestone.days * 24 * 60 * 60 * 1000);
      const msRemaining = nextMilestoneDate.getTime() - now.getTime();

      if (msRemaining <= 0) {
        setTimeUntilNext(null);
        return;
      }

      const days = Math.floor(msRemaining / (1000 * 60 * 60 * 24));
      const hours = Math.floor((msRemaining % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((msRemaining % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((msRemaining % (1000 * 60)) / 1000);

      setTimeUntilNext({
        days,
        hours,
        minutes,
        seconds,
        nextMilestone: nextMilestone.name,
      });
    };

    updateCountdown();
    const interval = setInterval(updateCountdown, 1000);

    return () => clearInterval(interval);
  }, [signalCreatedAt]);

  if (!timeUntilNext) {
    return null;
  }

  return (
    <Card className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-950/30 dark:to-cyan-950/30 border-blue-200 dark:border-blue-800">
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          {/* Left: Icon and Title */}
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500 rounded-lg">
              <Clock className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                Next Performance Update
              </h3>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                {timeUntilNext.nextMilestone} returns available in:
              </p>
            </div>
          </div>

          {/* Right: Countdown */}
          <div className="flex items-center gap-3">
            {timeUntilNext.days > 0 && (
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {timeUntilNext.days}
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">days</div>
              </div>
            )}
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {String(timeUntilNext.hours).padStart(2, '0')}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">hrs</div>
            </div>
            <div className="text-gray-400 text-2xl font-bold">:</div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {String(timeUntilNext.minutes).padStart(2, '0')}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">min</div>
            </div>
            <div className="text-gray-400 text-2xl font-bold">:</div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {String(timeUntilNext.seconds).padStart(2, '0')}
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">sec</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
