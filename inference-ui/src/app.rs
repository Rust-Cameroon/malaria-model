use crate::pages;
use yew::prelude::*;
use yew_router::prelude::*;

fn get_locale() -> String {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("locale").ok().flatten())
        .map(|v| {
            if v == "fr" {
                "fr".to_string()
            } else {
                "en".to_string()
            }
        })
        .unwrap_or_else(|| "en".to_string())
}

fn set_locale(value: &str) {
    if let Some(w) = web_sys::window() {
        if let Ok(Some(storage)) = w.local_storage() {
            let _ = storage.set_item("locale", value);
        }
    }
}

#[derive(Routable, PartialEq, Eq, Clone, Debug)]
pub enum Route {
    #[at("/")]
    Home,
    #[at("/demo")]
    Demo,
    #[at("/analyze")]
    Analyze,
    #[not_found]
    #[at("/404")]
    NotFound,
}

fn switch(route: Route) -> Html {
    match route {
        Route::Home => html! { <pages::home::HomePage /> },
        Route::Demo => html! { <pages::demo::DemoPage /> },
        Route::Analyze => html! { <pages::analyze::AnalyzePage /> },
        Route::NotFound => html! {
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
                <div class="bg-white/5 border border-white/10 rounded-xl p-6">
                    <h2 class="text-xl font-semibold">{"Page not found"}</h2>
                </div>
            </div>
        },
    }
}

#[derive(Properties, PartialEq)]
struct LayoutProps {
    #[prop_or_default]
    pub children: Children,
}

#[function_component(Layout)]
fn layout(props: &LayoutProps) -> Html {
    html! {
        <div class="relative min-h-screen font-sans text-white flex flex-col">
            <div aria-hidden="true" class="pointer-events-none absolute inset-0 -z-20 bg-cover bg-center bg-no-repeat" style="background-image:url('/static/malaria.jpeg')"></div>
            <div aria-hidden="true" class="pointer-events-none absolute inset-0 -z-10" style="background:linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.8))"></div>

            <header class="fixed top-0 inset-x-0 z-50 backdrop-blur supports-[backdrop-filter]:bg-black/30 bg-black/40">
                <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 grid grid-cols-3 items-center">
                    <div class="flex items-center">
                        { html!{
                            <Link<Route> to={Route::Home} classes="text-lg font-bold tracking-tight text-bright-blue">
                                { if get_locale() == "fr" { "Détection du paludisme" } else { "Malaria Detection" } }
                            </Link<Route>>
                        } }
                    </div>
                    <nav class="flex items-center justify-center gap-6 text-sm text-bright-blue opacity-90">
                        { html!{
                            <>
                                <Link<Route> to={Route::Demo} classes="hover:opacity-100">
                                    { if get_locale() == "fr" { "Démo" } else { "Demo" } }
                                </Link<Route>>
                                <Link<Route> to={Route::Analyze} classes="hover:opacity-100">
                                    { if get_locale() == "fr" { "Analyser" } else { "Analyze" } }
                                </Link<Route>>
                                <a href="#features" class="hover:opacity-100">
                                    { if get_locale() == "fr" { "Fonctionnalités" } else { "Features" } }
                                </a>
                                <a href="#contact" class="hover:opacity-100">
                                    {"Contact"}
                                </a>
                            </>
                        } }
                    </nav>
                    <div class="flex items-center justify-end">
                        <button aria-label="Change language" title="Change language" class="text-bright-blue hover:opacity-100 opacity-90 p-1 rounded-full"
                            onclick={Callback::from(|_| {
                                let l = get_locale();
                                if l == "fr" { set_locale("en"); } else { set_locale("fr"); }
                                if let Some(w) = web_sys::window() { let _ = w.location().reload(); }
                            })}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="w-7 h-7">
                                <circle cx="12" cy="12" r="9" />
                                <path d="M3 12h18" />
                                <path d="M12 3a15.3 15.3 0 0 1 4 9 15.3 15.3 0 0 1-4 9 15.3 15.3 0 0 1-4-9 15.3 15.3 0 0 1 4-9z" />
                            </svg>
                        </button>
                    </div>
                </div>
            </header>

            <main class="pt-16 flex flex-col items-center flex-1">
                { for props.children.iter() }
            </main>
            <div id="portal-container"></div>
            <footer class="mt-8 border-t border-white/10 backdrop-blur supports-[backdrop-filter]:bg-black/20 bg-black/20">
                <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5 grid grid-cols-3 items-start text-xs text-white/80">
                    <div>
                        <div class="font-semibold tracking-tight">
                            { if get_locale() == "fr" { "Détection du paludisme" } else { "Malaria Detection" } }
                        </div>
                        <div class="opacity-70">
                            { if get_locale() == "fr" { "Analyse assistée des frottis (recherche et éducation)" } else { "AI assisted smear analysis (research and education)" } }
                        </div>
                    </div>
                    <nav class="flex items-center justify-center gap-6">
                        <a href="#confidentiality" class="hover:opacity-100 opacity-80">
                            { if get_locale() == "fr" { "Confidentialité" } else { "Confidentiality" } }
                        </a>
                        <a href="#terms" class="hover:opacity-100 opacity-80">
                            { if get_locale() == "fr" { "Conditions" } else { "Terms" } }
                        </a>
                        <a href="#contact" class="hover:opacity-100 opacity-80">
                            {"Contact"}
                        </a>
                    </nav>
                    <div class="flex items-center justify-end opacity-70">
                        { if get_locale() == "fr" { "© 2026 Tous droits réservés." } else { "© 2026 All rights reserved." } }
                    </div>
                </div>
            </footer>
        </div>
    }
}

#[function_component(App)]
pub fn app() -> Html {
    html! {
        <BrowserRouter>
            <Layout>
                <Switch<Route> render={switch} />
            </Layout>
        </BrowserRouter>
    }
}
