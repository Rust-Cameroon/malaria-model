use yew::prelude::*;

fn get_locale() -> String {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("locale").ok().flatten())
        .unwrap_or_else(|| "en".to_string())
}

#[function_component(FeaturesPage)]
pub fn features_page() -> Html {
    let locale = get_locale();
    let is_fr = locale == "fr";

    html! {
        <div class="w-full min-h-screen bg-black/20">
            <div class="max-w-7xl mx-auto px-6 sm:px-8 py-20">
                <div class="text-center mb-16">
                    <h1 class="text-4xl md:text-5xl font-extrabold tracking-tight mb-4">
                        { if is_fr { "Fonctionnalités de la Plateforme" } else { "Platform Capabilities" } }
                    </h1>
                    <p class="text-lg opacity-80 max-w-2xl mx-auto">
                        { if is_fr {
                            "Découvrez comment Giemsa AI utilise la vision par ordinateur avancée pour transformer le diagnostic du paludisme."
                        } else {
                            "Discover how Giemsa AI leverages advanced computer vision to transform malaria diagnostics."
                        } }
                    </p>
                </div>

                <div class="grid md:grid-cols-2 gap-12 mb-20">
                    <div class="space-y-6">
                        <div class="h-12 w-12 rounded-lg bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                             <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>
                        </div>
                        <h3 class="text-2xl font-bold">
                            { if is_fr { "Détection de Précision" } else { "Precision Detection" } }
                        </h3>
                        <p class="opacity-80 leading-relaxed">
                            { if is_fr {
                                "Notre modèle est entraîné sur des milliers d'images de frottis sanguins colorés au Giemsa. Il identifie les parasites Plasmodium avec une grande sensibilité, distinguant les érythrocytes infectés des sains, même dans les échantillons à faible parasitémie."
                            } else {
                                "Our core model is trained on thousands of Giemsa-stained thin blood smear images. It identifies Plasmodium parasites with high sensitivity, distinguishing between infected and uninfected erythrocytes even in low-parasitemia samples."
                            } }
                        </p>
                        <ul class="space-y-3 opacity-80">
                            <li class="flex items-center gap-2">
                                <span class="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                                { if is_fr { "Plus de 96% de précision sur les ensembles de validation" } else { "96%+ detection accuracy on validation sets" } }
                            </li>
                            <li class="flex items-center gap-2">
                                <span class="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                                { if is_fr { "Robuste face aux variations de coloration" } else { "Robust against staining variations" } }
                            </li>
                        </ul>
                    </div>
                    <div class="rounded-2xl border border-white/10 bg-white/5 p-8 flex items-center justify-center">
                        <div class="relative">
                            <div class="absolute -inset-4 bg-emerald-500/20 blur-xl rounded-full"></div>
                             <img src="/static/smear_illustration.png" class="relative rounded-lg shadow-2xl w-64 h-64 object-cover" alt="Detection Example" />
                        </div>
                    </div>
                </div>

                <div class="grid md:grid-cols-2 gap-12 mb-20">
                    <div class="order-2 md:order-1 rounded-2xl border border-white/10 bg-white/5 p-8 flex items-center justify-center">
                        <div class="grid grid-cols-2 gap-4 w-full max-w-sm">
                            <div class="p-4 rounded-xl bg-black/40 border border-white/5 text-center">
                                <div class="text-2xl font-bold text-emerald-400">{"< 2s"}</div>
                                <div class="text-xs opacity-60">{ if is_fr { "Temps d'inférence" } else { "Inference Time" } }</div>
                            </div>
                            <div class="p-4 rounded-xl bg-black/40 border border-white/5 text-center">
                                <div class="text-2xl font-bold text-emerald-400">{"4"}</div>
                                <div class="text-xs opacity-60">{ if is_fr { "Classes d'espèces" } else { "Species Classes" } }</div>
                            </div>
                            <div class="p-4 rounded-xl bg-black/40 border border-white/5 text-center">
                                <div class="text-2xl font-bold text-emerald-400">{"API"}</div>
                                <div class="text-xs opacity-60">{ if is_fr { "Intégration Standard" } else { "Standard Integration" } }</div>
                            </div>
                            <div class="p-4 rounded-xl bg-black/40 border border-white/5 text-center">
                                <div class="text-2xl font-bold text-emerald-400">{"PDF"}</div>
                                <div class="text-xs opacity-60">{ if is_fr { "Rapport Auto" } else { "Auto Reporting" } }</div>
                            </div>
                        </div>
                    </div>
                    <div class="order-1 md:order-2 space-y-6">
                        <div class="h-12 w-12 rounded-lg bg-blue-500/20 flex items-center justify-center text-blue-400">
                             <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                        </div>
                        <h3 class="text-2xl font-bold">
                            { if is_fr { "Optimisation du Flux de Travail" } else { "Workflow Optimization" } }
                        </h3>
                        <p class="opacity-80 leading-relaxed">
                            { if is_fr {
                                "Giemsa AI n'est pas seulement un modèle ; c'est un outil de flux de travail. En automatisant le dépistage initial, nous permettons aux techniciens de se concentrer uniquement sur les cas suspects."
                            } else {
                                "Giemsa AI isn't just a model; it's a workflow tool. By automating the initial screening, we allow lab technicians to focus only on the suspect cases flagged by the system."
                            } }
                        </p>
                         <ul class="space-y-3 opacity-80">
                            <li class="flex items-center gap-2">
                                <span class="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                                { if is_fr { "Réduit le temps de microscopie manuelle de ~40%" } else { "Reduces manual microscopy time by ~40%" } }
                            </li>
                            <li class="flex items-center gap-2">
                                <span class="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                                { if is_fr { "Rapport PDF standardisé pour les dossiers" } else { "Standardized PDF reporting for patient records" } }
                            </li>
                        </ul>
                    </div>
                </div>

                 <div class="pt-20 border-t border-white/10 text-center">
                    <h2 class="text-2xl font-bold mb-6">
                        { if is_fr { "Prêt à le voir en action ?" } else { "Ready to see it in action?" } }
                    </h2>
                    <a href="/demo" class="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition-colors">
                        { if is_fr { "Essayez la Démo Interactive" } else { "Try the Interactive Demo" } }
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
                    </a>
                </div>
            </div>
        </div>
    }
}
