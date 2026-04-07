import logoUrl from "../assets/logo_white.png";
import { useTheme } from "../context/ThemeContext";

export default function FoleoLogo({ size = "md", className = "" }) {
  const { theme } = useTheme();
  const sizeClasses = {
    sm: "h-7",
    md: "h-9",
    lg: "h-12",
  };
  const isLight = theme === "light";

  return (
    <div className={`flex items-center ${className}`}>
      <img 
        src={logoUrl} 
        alt="Foleo Logo" 
        className={`${sizeClasses[size]} object-contain transition-all duration-500`}
        style={isLight ? { filter: "invert(1)" } : undefined}
      />
    </div>
  );
}
