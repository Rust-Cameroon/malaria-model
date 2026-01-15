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
                <div class="absolute inset-0 -z-10 bg-gradient-to-b from-emerald-700/20 via-transparent to-transparent"></div>
                <div class="max-w-7xl mx-auto px-6 sm:px-8 py-20 md:py-28 text-center">
                    <>
                        <div class="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs opacity-80">
                            { if get_locale() == "fr" { "Recherche et éducation" } else { "Research and Education Overview" } }
                        </div>
                        <h1 class="mt-4 text-4xl md:text-6xl font-extrabold tracking-tight">
                            { if get_locale() == "fr" { "Analyse IA des frottis du paludisme" } else { "AI analysis of malaria smears" } }
                            <span class="block text-emerald-400">
                                { if get_locale() == "fr" { "pour un triage clinique plus rapide" } else { "for faster clinical triage" } }
                            </span>
                        </h1>
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

            <section id="features" class="px-6 sm:px-8 py-16">
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

            <section class="px-6 sm:px-8 pb-20">
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
