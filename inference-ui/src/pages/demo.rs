use gloo_timers::future::TimeoutFuture;
use js_sys;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys;
use yew::prelude::*;

#[derive(Clone, PartialEq, Debug)]
struct DemoResult {
    parasite_detected: bool,
    confidence: f64,
    processed_at: String,
}

#[derive(Properties, PartialEq)]
struct StepProps {
    number: u8,
    title: AttrValue,
    #[prop_or_default]
    children: Children,
}

#[function_component(Step)]
fn step(props: &StepProps) -> Html {
    html! {
        <section class="rounded-2xl border border-white/20 p-6 bg-white/5 backdrop-blur-sm">
            <div class="flex items-center gap-3 mb-2">
                <div class="h-8 w-8 rounded-full bg-emerald-600 text-white flex items-center justify-center text-sm font-semibold">{props.number}</div>
                <h2 class="text-xl font-semibold">{props.title.clone()}</h2>
            </div>
            <div class="opacity-90 text-sm">{ for props.children.iter() }</div>
        </section>
    }
}

fn get_locale() -> String {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("locale").ok().flatten())
        .unwrap_or_else(|| "en".to_string())
}

#[function_component(DemoPage)]
pub fn demo_page() -> Html {
    let preview = use_state(|| None as Option<String>);
    let progress = use_state(|| 0u32);
    let running = use_state(|| false);
    let result = use_state(|| None as Option<DemoResult>);
    let locale = get_locale();
    let is_fr = locale == "fr";

    let choose_sample = {
        let preview = preview.clone();
        let result = result.clone();
        let progress = progress.clone();
        let running = running.clone();
        Callback::from(move |url: String| {
            preview.set(Some(url));
            result.set(None);
            progress.set(0);
            running.set(false);
        })
    };

    let on_analyze = {
        let running_state = running.clone();
        let progress_state = progress.clone();
        let result_state = result.clone();
        let preview_state = preview.clone();
        Callback::from(move |_| {
            if preview_state.is_none() || *running_state {
                return;
            }
            running_state.set(true);
            result_state.set(None);
            progress_state.set(0);

            let preview_val = (*preview_state).clone().unwrap();
            let progress_state = progress_state.clone();
            let running_state = running_state.clone();
            let result_state = result_state.clone();
            spawn_local(async move {
                let mut val = 0u32;
                while val < 100 {
                    let step = 5.max(js_sys::Math::floor(js_sys::Math::random() * 15.0) as u32);
                    val = (val + step).min(100);
                    progress_state.set(val);
                    TimeoutFuture::new(120).await;
                }
                progress_state.set(100);

                // Deterministic mapping for known samples (extension-agnostic)
                let p = preview_val.to_lowercase();
                let (parasite_detected, confidence) = if p.contains("infected") {
                    (true, 0.96)
                } else if p.contains("uninfected") {
                    (false, 0.98)
                } else {
                    // Fallback heuristic
                    let mut hash: i32 = 0;
                    for ch in preview_val.chars() {
                        hash += ch as i32;
                    }
                    let pd = hash % 3 != 0;
                    let conf = (0.7 + ((hash.rem_euclid(30) as f64) / 100.0)).min(0.99);
                    (pd, conf)
                };

                result_state.set(Some(DemoResult {
                    parasite_detected,
                    confidence,
                    processed_at: js_sys::Date::new_0().to_iso_string().into(),
                }));
                running_state.set(false);
            });
        })
    };

    let samples = [
        (
            "/static/demo_infected.png",
            if is_fr { "Infecté" } else { "Infected" },
        ),
        (
            "/static/uninfected.png",
            if is_fr {
                "Non infecté"
            } else {
                "Not infected"
            },
        ),
    ];

    html! {
        <div class="w-full max-w-7xl mx-auto px-6 sm:px-8 py-20 md:py-24 space-y-6">
            <header class="text-center mb-2">
                <h1 class="text-3xl md:text-4xl font-extrabold tracking-tight">
                    { if is_fr { "Démo : Analyser un frottis sanguin" } else { "Demo: Analyzing a blood smear image" } }
                </h1>
                <p class="mt-2 text-sm opacity-80">
                    { if is_fr { "Suivez les étapes ci-dessous pour voir comment l'analyse fonctionne. Elle s'exécute localement dans votre navigateur." } else { "Follow the steps below to preview how the analysis works. It runs locally in your browser." } }
                </p>
            </header>

            <Step number={1} title={ if is_fr { "Aperçu" } else { "Preview" } }>
                { if is_fr { "Suivez les étapes guidées pour voir comment l'analyse fonctionne avec des images exemples." } else { "Follow the guided steps to see how the analysis works using example images." } }
            </Step>

            <div class="grid lg:grid-cols-2 gap-6">
                <Step number={2} title={ if is_fr { "Sélectionner une image" } else { "Select a sample image" } }>
                    <div class="grid grid-cols-3 gap-3 mb-4">
                        { for samples.iter().map(|(url, label)| {
                            let choose_sample = choose_sample.clone();
                            let url_str = (*url).to_string();
                            html! {
                                <button onclick={Callback::from(move |_| choose_sample.emit(url_str.clone()))} class="group relative rounded-lg overflow-hidden border border-white/10">
                                    <img src={*url} alt={*label} class="h-24 w-full object-cover group-hover:opacity-90" />
                                    <span class="absolute bottom-1 left-1 text-[10px] bg-black/50 px-1 rounded">{*label}</span>
                                </button>
                            }
                        }) }
                    </div>
                    if let Some(p) = (*preview).clone() {
                        <img src={p} alt="preview" class="mt-3 rounded-lg max-h-64 object-contain border border-white/10" />
                    }
                </Step>

                <Step number={3} title={ if is_fr { "Analyser" } else { "Analyze" } }>
                    <p>{ if is_fr { "Sélectionner une image échantillon" } else { "Select a sample image" } }</p>
                    <button onclick={on_analyze}
                        disabled={(*preview).is_none() || *running}
                        class={classes!(
                            "mt-4", "px-4", "py-2", "rounded-md", "text-white",
                            if (*preview).is_none() || *running { Some("bg-gray-500/60") } else { Some("bg-emerald-600 hover:bg-emerald-700") }
                        )}
                    >{ if *running { if is_fr { "Analyse en cours..." } else { "Analyzing..." } } else if is_fr { "Analyser" } else { "Analyze" } }</button>
                    <p class="mt-2 text-[11px] opacity-60">
                        { if is_fr { "Démo simulée côté client. Non destinée à un usage diagnostique." } else { "Simulated demo on the client side. Not intended for diagnostic use." } }
                    </p>
                    { if *running || *progress > 0 {
                        html!{
                            <div class="mt-4">
                                <div class="h-2 bg-white/10 rounded">
                                    <div class="h-2 bg-emerald-500 rounded" style={format!("width: {}%; transition: width 0.2s", *progress)} />
                                </div>
                                <div class="text-xs mt-1 opacity-80">{format!("{}%", *progress)}</div>
                            </div>
                        }
                    } else { html!{} } }
                </Step>
            </div>

            <Step number={4} title={ if is_fr { "Voir les résultats" } else { "View results" } }>
                { if let Some(r) = (*result).clone() {
                    html!{
                        <div class="space-y-2">
                            <div>
                                { if is_fr { "Parasite détecté : " } else { "Parasite detected: " } }
                                { if r.parasite_detected { html!{ <span class="text-red-400 font-semibold">{ if is_fr { "Oui" } else { "Yes" } }</span> } } else { html!{ <span class="text-emerald-400 font-semibold">{ if is_fr { "Non" } else { "No" } }</span> } } }
                            </div>
                            <div>{format!("{}: {:.1}%", if is_fr { "Confiance" } else { "Confidence" }, r.confidence * 100.0)}</div>
                            <div class="text-xs opacity-60 mb-3">{format!("{}: {}", if is_fr { "Traité à" } else { "Processed at" }, r.processed_at)}</div>
                            <button
                                onclick={
                                    let r = r.clone();
                                    let preview = (*preview).clone();
                                    Callback::from(move |_| {
                                        let date = js_sys::Date::new_0().to_locale_string("en-US", &wasm_bindgen::JsValue::from_str("{\"dateStyle\":\"long\", \"timeStyle\":\"short\"}")).as_string().unwrap_or_default();
                                        let data = serde_json::json!({
                                            "infected": r.parasite_detected,
                                            "species": if r.parasite_detected { "Detected (Demo)" } else { "Uninfected (Demo)" },
                                            "speciesProb": r.confidence,
                                            "stage": "Not specified (Demo)",
                                            "stageProb": 0.0,
                                            "date": date,
                                            "imageUrl": preview
                                        });

                                        if let Ok(js_data) = serde_wasm_bindgen::to_value(&data) {
                                            let window = web_sys::window().unwrap();
                                            let _ = js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("generateMalariaReport"))
                                                .unwrap()
                                                .dyn_into::<js_sys::Function>()
                                                .unwrap()
                                                .call1(&wasm_bindgen::JsValue::NULL, &js_data);
                                        }
                                    })
                                }
                                class="bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded transition-colors flex items-center gap-2 text-xs"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                                { if is_fr { "Télécharger le rapport (PDF)" } else { "Download Demo Report (PDF)" } }
                            </button>
                        </div>
                    }
                } else {
                    html!{ <p class="text-sm opacity-80">{ if is_fr { "Aucun résultat. Complétez les étapes 1 à 3 et cliquez sur Analyser." } else { "No results yet. Complete steps 1 to 3 and click Analyze." } }</p> }
                }}
            </Step>
        </div>
    }
}
