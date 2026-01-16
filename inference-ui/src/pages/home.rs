use yew::prelude::*;

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

#[function_component(HomePage)]
pub fn home_page() -> Html {
    html! {
        <div class="w-full">
            <section class="relative overflow-hidden">
                <nav class="absolute top-0 left-0 w-full z-10 px-6 py-6 flex items-center justify-between">
                    <div class="flex items-center gap-3">
                        <img src="/static/logo.png" class="h-8 w-8 rounded-full" alt="Giemsa AI Logo" />
                        <span class="font-bold text-xl tracking-tight">{"Giemsa AI"}</span>
                    </div>
                </nav>
                <div class="absolute inset-0 -z-10 bg-gradient-to-b from-emerald-900/30 via-transparent to-transparent"></div>
                <div class="max-w-7xl mx-auto px-6 sm:px-8 py-24 md:py-32 text-center">
                    <>
                        <div class="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs opacity-80">
                            { if get_locale() == "fr" { "Recherche et éducation" } else { "Research and Education Overview" } }
                        </div>
                        <h1 class="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight text-white mb-2">
                            {"Giemsa AI"}
                        </h1>
                        <h2 class="text-2xl md:text-4xl font-bold tracking-tight opacity-90">
                            { if get_locale() == "fr" { "Analyse IA des frottis du paludisme" } else { "AI analysis of malaria smears" } }
                            <span class="block text-emerald-400">
                                { if get_locale() == "fr" { "pour un triage clinique plus rapide" } else { "for faster clinical triage" } }
                            </span>
                        </h2>
                        <p class="mt-4 text-base md:text-lg opacity-80 max-w-3xl mx-auto">
                            { if get_locale() == "fr" {
                                "Vision par ordinateur moderne pour aider cliniciens et laboratoires avec une évaluation rapide et cohérente des images de frottis sanguins."
                            } else {
                                "Modern computer vision to assist clinicians and laboratories with rapid and consistent evaluations of blood smear images."
                            } }
                        </p>
                        <div class="mt-8 flex items-center justify-center gap-3">
                            <a href="/demo" class="px-5 py-2.5 rounded-md bg-emerald-600 text-white hover:bg-emerald-700">
                                { if get_locale() == "fr" { "Voir la démonstration" } else { "View the demo stream" } }
                            </a>
                            <a href="#features" class="px-5 py-2.5 rounded-md border border-white/20 hover:bg-white/5">
                                { if get_locale() == "fr" { "En savoir plus" } else { "Learn more" } }
                            </a>
                        </div>
                        <p class="mt-3 text-xs opacity-60">
                            { if get_locale() == "fr" { "Dispositif non médical. À des fins de recherche et d'évaluation éducative uniquement." } else { "Not a medical device. For research and educational evaluation purposes only." } }
                        </p>
                    </>
                    <div class="mt-12 grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-4xl mx-auto">
                        <>
                            <div class="rounded-lg border border-white/10 p-4">
                                <div class="text-2xl font-bold">
                                    { if get_locale() == "fr" { "Instantané" } else { "Instant" } }
                                </div>
                                <div class="text-xs opacity-70">
                                    { if get_locale() == "fr" { "Analyse côté client" } else { "Customer‑side overview analysis" } }
                                </div>
                            </div>
                            <div class="rounded-lg border border-white/10 p-4">
                                <div class="text-2xl font-bold">
                                    { if get_locale() == "fr" { "Cohérent" } else { "Consistent" } }
                                </div>
                                <div class="text-xs opacity-70">
                                    { if get_locale() == "fr" { "Évaluations guidées par le modèle" } else { "Model‑driven assessments" } }
                                </div>
                            </div>
                            <div class="rounded-lg border border-white/10 p-4">
                                <div class="text-2xl font-bold">
                                    { if get_locale() == "fr" { "Évolutif" } else { "Scalable" } }
                                </div>
                                <div class="text-xs opacity-70">
                                    { if get_locale() == "fr" { "Conçu pour des débits élevés" } else { "Designed for high flow rate" } }
                                </div>
                            </div>
                        </>
                    </div>
                </div>
            </section>

            <section class="px-6 sm:px-8 py-12 bg-white/[0.02] border-t border-white/5">
                <div class="max-w-7xl mx-auto">
                    <div class="text-center mb-12">
                        <h2 class="text-3xl md:text-4xl font-bold mb-4">
                            { if get_locale() == "fr" { "L'Urgence du Paludisme en Afrique" } else { "The Malaria Crisis in Africa" } }
                        </h2>
                        <p class="text-lg opacity-80 max-w-3xl mx-auto">
                            { if get_locale() == "fr" {
                                "Le paludisme reste l'un des plus grands défis de santé publique sur le continent. Voici pourquoi notre solution est cruciale."
                            } else {
                                "Malaria remains one of the greatest public health challenges on the continent. Here is why our solution is crucial."
                            } }
                        </p>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                        <div class="p-6 rounded-2xl bg-emerald-950/30 border border-emerald-500/20 text-center animate-fade-in" style="animation-delay: 0.1s">
                            <div class="text-4xl font-black text-emerald-400 mb-2">{"265M"}</div>
                            <p class="text-sm font-semibold opacity-90 uppercase tracking-wider">
                                { if get_locale() == "fr" { "Cas Annuels en Afrique" } else { "Annual Cases in Africa" } }
                            </p>
                            <p class="mt-2 text-xs opacity-60">{"95% of the total global burden falls on the African continent."}</p>
                        </div>
                        <div class="p-6 rounded-2xl bg-emerald-950/30 border border-emerald-500/20 text-center animate-fade-in" style="animation-delay: 0.2s">
                            <div class="text-4xl font-black text-emerald-400 mb-2">{"579K"}</div>
                            <p class="text-sm font-semibold opacity-90 uppercase tracking-wider">
                                { if get_locale() == "fr" { "Décès Annuels" } else { "Annual Deaths" } }
                            </p>
                            <p class="mt-2 text-xs opacity-60">{"The vast majority of malaria-related deaths occur in the WHO African Region."}</p>
                        </div>
                        <div class="p-6 rounded-2xl bg-emerald-950/30 border border-emerald-500/20 text-center animate-fade-in" style="animation-delay: 0.3s">
                            <div class="text-4xl font-black text-emerald-400 mb-2">{"76%"}</div>
                            <p class="text-sm font-semibold opacity-90 uppercase tracking-wider">
                                { if get_locale() == "fr" { "Enfants < 5 ans" } else { "Children < 5 years" } }
                            </p>
                            <p class="mt-2 text-xs opacity-60">{"The most vulnerable demographic, accounting for 3 out of every 4 malaria deaths."}</p>
                        </div>
                        <div class="p-6 rounded-2xl bg-emerald-950/30 border border-emerald-500/20 text-center animate-fade-in" style="animation-delay: 0.4s">
                            <div class="text-4xl font-black text-emerald-400 mb-2">{"$5.1B"}</div>
                            <p class="text-sm font-semibold opacity-90 uppercase tracking-wider">
                                { if get_locale() == "fr" { "Déficit de Financement" } else { "Funding Gap" } }
                            </p>
                            <p class="mt-2 text-xs opacity-60">{"The gap between current resources and the US$9.3B needed for elimination by 2025."}</p>
                        </div>
                    </div>
                </div>
            </section>

            <section class="px-6 sm:px-8 py-16 overflow-hidden">
                <div class="max-w-7xl mx-auto flex flex-col lg:flex-row items-center gap-12">
                    <div class="flex-1 space-y-6">
                        <div class="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-400">
                             { if get_locale() == "fr" { "Aperçu des Données" } else { "Data Insight" } }
                        </div>
                        <h2 class="text-3xl md:text-5xl font-bold leading-tight">
                            { if get_locale() == "fr" { "Pourquoi cette plateforme est nécessaire ?" } else { "Why is this platform necessary?" } }
                        </h2>
                        <div class="space-y-4 text-lg opacity-80">
                            <p>
                                { if get_locale() == "fr" {
                                    "La détection précoce est la clé pour réduire la mortalité. Cependant, de nombreux établissements de santé en Afrique manquent de microscopistes experts ou font face à une charge de travail écrasante."
                                } else {
                                    "Early detection is key to reducing mortality. However, many health facilities in Africa lack expert microscopists or face overwhelming workloads."
                                } }
                            </p>
                            <p>
                                { if get_locale() == "fr" {
                                    "Notre plateforme d'IA comble ce fossé en fournissant des outils d'analyse automatisés, rapides et précis pour soutenir le diagnostic clinique dans les régions les plus touchées."
                                } else {
                                    "Our AI platform bridges this gap by providing automated, rapid, and accurate analysis tools to support clinical diagnosis in the most affected regions."
                                } }
                            </p>
                        </div>
                        <div class="pt-4 border-t border-white/10 grid grid-cols-2 gap-4">
                            <div>
                                <h4 class="font-bold text-emerald-400 text-sm italic">{"High Burden Zone 1"}</h4>
                                <p class="text-xl font-bold">{"Nigeria (31.9%)"}</p>
                                <p class="text-xs opacity-60">{"of global malaria deaths"}</p>
                            </div>
                            <div>
                                <h4 class="font-bold text-emerald-400 text-sm italic">{"High Burden Zone 2"}</h4>
                                <p class="text-xl font-bold">{"DRC (11.7%)"}</p>
                                <p class="text-xs opacity-60">{"of global malaria deaths"}</p>
                            </div>
                        </div>
                    </div>
                    <div class="flex-1 relative">
                        <div class="absolute -inset-4 bg-emerald-500/20 blur-3xl -z-10 rounded-full"></div>
                        <img
                            src="/static/malaria_map_burden.png"
                            alt="Malaria Burden Map in Africa"
                            class="rounded-2xl border border-white/10 shadow-2xl shadow-emerald-900/20 w-full animate-scale-in"
                        />
                    </div>
                </div>
            </section>


            <section id="features" class="px-6 sm:px-8 py-12">
                <div class="max-w-7xl mx-auto grid md:grid-cols-3 gap-6">
                    <article class="rounded-xl border border-white/10 p-6 bg-white/5">
                        <div class="flex items-center gap-3 mb-3">
                            <div class="h-5 w-5 text-emerald-400">{""}</div>
                            <h3 class="font-semibold">{"Faster triage"}</h3>
                        </div>
                        <p class="text-sm opacity-80">{"Augment manual microscopy workflows with quick automatic screening to assist operators."}</p>
                    </article>
                    <article class="rounded-xl border border-white/10 p-6 bg-white/5">
                        <div class="flex items-center gap-3 mb-3">
                            <div class="h-5 w-5 text-emerald-400">{""}</div>
                            <h3 class="font-semibold">{"Lab workflow"}</h3>
                        </div>
                        <p class="text-sm opacity-80">{"Integrate with a Rust API providing health checks and prediction endpoints."}</p>
                    </article>
                    <article class="rounded-xl border border-white/10 p-6 bg-white/5">
                        <div class="flex items-center gap-3 mb-3">
                            <div class="h-5 w-5 text-emerald-400">{""}</div>
                            <h3 class="font-semibold">{"Education"}</h3>
                        </div>
                        <p class="text-sm opacity-80">{"Use the demo to teach image-based diagnostics and model probability interpretation."}</p>
                    </article>
                </div>
            </section>

            <section class="px-6 sm:px-8 pb-16">
                <div class="max-w-5xl mx-auto rounded-2xl border border-white/10 p-8 bg-white/5 text-center">
                    <div class="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs opacity-80 mb-3">{"Commitment"}</div>
                    <h3 class="text-2xl md:text-3xl font-bold">{"Responsible AI for healthcare"}</h3>
                    <p class="mt-2 text-sm opacity-80 max-w-3xl mx-auto">{"We aim for transparent models and practical demos that promote safe adoption of AI tools."}</p>
                    <div class="mt-6 flex items-center justify-center gap-3">
                        <a href="mailto:inforustcameroon@gmail.com" class="px-5 py-2.5 rounded-md bg-emerald-600 text-white hover:bg-emerald-700">{"Partner with us"}</a>
                        <a href="/demo" class="px-5 py-2.5 rounded-md border border-white/20 hover:bg-white/5">{"Explore examples"}</a>
                    </div>
                    <div class="mt-4 flex items-center justify-center gap-6 text-xs opacity-70">
                        <div class="flex items-center gap-2">{"Clinical"}</div>
                        <div class="flex items-center gap-2">{"Lab"}</div>
                        <div class="flex items-center gap-2">{"On-device"}</div>
                    </div>
                </div>
            </section>

            <section id="contact" class="px-6 sm:px-8 pb-24">
                <div class="max-w-4xl mx-auto text-center">
                    <h3 class="text-2xl md:text-3xl font-bold">{"Get in touch"}</h3>
                    <p class="mt-2 text-sm opacity-80">{"Questions or collaboration opportunities? We'd love to hear from you."}</p>
                    <div class="mt-6">
                        <a href="mailto:inforustcameroon@gmail.com" class="inline-flex px-5 py-2.5 rounded-md bg-emerald-600 text-white hover:bg-emerald-700">{"Email us"}</a>
                    </div>
                </div>
            </section>
        </div>
    }
}
