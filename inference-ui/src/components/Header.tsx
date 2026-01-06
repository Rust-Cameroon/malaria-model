import * as React from "react";
import { LanguageSwitcher } from "./LanguageSwitcher";
import { buttonVariants } from "./ui/buttonVariants";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";

export const Header: React.FC = () => {
  const { t } = useTranslation();
  return (
    <header className="fixed top-0 inset-x-0 z-50 backdrop-blur supports-[backdrop-filter]:bg-black/30 bg-black/40 text-bright-blue">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link to="/" className="text-lg font-bold tracking-tight">{t("app.title")}</Link>
        </div>
        <nav className="flex items-center gap-4 opacity-90">
          <Link to="/demo" className={buttonVariants({ variant: "ghost" })}>{t("app.nav.demo")}</Link>
          <Link to="/analyze" className={buttonVariants({ variant: "ghost" })}>Analyze</Link>
          <a href="#features" className={buttonVariants({ variant: "ghost" })}>{t("app.nav.features")}</a>
          <a href="mailto:contact@yourdomain.com" className={buttonVariants({ variant: "ghost" })}>{t("app.nav.contact")}</a>
        </nav>
        <div className="flex items-center gap-2">
          <LanguageSwitcher />
        </div>
      </div>
    </header>
  );
};
