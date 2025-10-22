'use client';

interface HighlightedTextProps {
  text: string;
  searchQuery: string;
}

export function HighlightedText({ text, searchQuery }: HighlightedTextProps) {
  if (!searchQuery || !text) {
    return <>{text}</>;
  }

  const regex = new RegExp(`(${searchQuery.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const parts = text.split(regex);

  return (
    <>
      {parts.map((part, index) => {
        if (regex.test(part)) {
          return (
            <mark
              key={index}
              className="bg-gradient-to-r from-[#001F3F]/20 to-[#00AEEF]/20 text-gray-900 dark:text-gray-100 font-semibold px-0.5 rounded"
            >
              {part}
            </mark>
          );
        }
        return <span key={index}>{part}</span>;
      })}
    </>
  );
}
