use gloo_net::http::Request;
use js_sys;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::spawn_local;
use web_sys::{window, DragEvent, File as WebFile, FormData, HtmlInputElement, Url};
use yew::prelude::*;

#[derive(Clone, PartialEq, serde::Deserialize, serde::Serialize, Debug)]
pub struct PredictResponse {
    pub infected: bool,
    pub predicted_species: String,
    pub species_probabilities: Vec<f64>,
    pub stage_probabilities: [f64; 4],
}

fn pct(v: f64) -> i32 {
    (v * 100.0).round().clamp(0.0, 100.0) as i32
}

fn argmax(v: &[f64]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

fn api_base() -> String {
    let w = window().unwrap();
    // 1) If hosted on Render, point to the Render API domain
    if let Ok(loc) = w.location().host() {
        if loc.ends_with("onrender.com") {
            return "https://malaria-inference-api.onrender.com".to_string();
        }
    }
    // 2) Allow runtime override via a global variable (can be injected by an inline script)
    if let Ok(Some(val)) =
        js_sys::Reflect::get(&w, &JsValue::from_str("VITE_API_BASE")).map(|v| v.as_string())
    {
        let trimmed = val.trim_end_matches('/').to_string();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }
    // 3) Fallback to local development API
    "http://localhost:8080".to_string()
}

#[function_component(AnalyzePage)]
pub fn analyze_page() -> Html {
    let server_ok = use_state(|| None as Option<bool>);
    let file = use_state(|| None as Option<WebFile>);
    let preview_url = use_state(|| None as Option<String>);
    let loading = use_state(|| false);
    let error = use_state(|| None as Option<String>);
    let result = use_state(|| None as Option<PredictResponse>);

    {
        let server_ok = server_ok.clone();
        use_effect_with((), move |_| {
            let server_ok = server_ok.clone();
            spawn_local(async move {
                let url = format!("{}/health", api_base());
                match Request::get(&url).send().await {
                    Ok(resp) if resp.ok() => server_ok.set(Some(true)),
                    _ => server_ok.set(Some(false)),
                }
            });
            || {}
        });
    }

    let on_file_change = {
        let file = file.clone();
        let preview_url_state = preview_url.clone();
        let result = result.clone();
        let error_state = error.clone();
        Callback::from(move |e: Event| {
            let input: HtmlInputElement = e.target_unchecked_into();
            if let Some(list) = input.files() {
                if let Some(f) = list.get(0) {
                    result.set(None);
                    error_state.set(None);
                    if let Some(prev) = (*preview_url_state).clone() {
                        let _ = Url::revoke_object_url(&prev);
                    }
                    let url = Url::create_object_url_with_blob(&f).unwrap_or_default();
                    preview_url_state.set(Some(url));
                    file.set(Some(f));
                }
            }
        })
    };

    let on_drop = {
        let file = file.clone();
        let preview_url_state = preview_url.clone();
        let result = result.clone();
        let error_state = error.clone();
        Callback::from(move |e: DragEvent| {
            e.prevent_default();
            if let Some(dt) = e.data_transfer() {
                if let Some(list) = dt.files() {
                    if let Some(f) = list.get(0) {
                        result.set(None);
                        error_state.set(None);
                        if let Some(prev) = (*preview_url_state).clone() {
                            let _ = Url::revoke_object_url(&prev);
                        }
                        let url = Url::create_object_url_with_blob(&f).unwrap_or_default();
                        preview_url_state.set(Some(url));
                        file.set(Some(f));
                    }
                }
            }
        })
    };

    let on_drag_over = Callback::from(|e: DragEvent| e.prevent_default());

    let on_analyze = {
        let loading_outer = loading.clone();
        let error_outer = error.clone();
        let result_outer = result.clone();
        let file_outer = file.clone();
        Callback::from(move |_| {
            let loading = loading_outer.clone();
            let error_state = error_outer.clone();
            let result_state = result_outer.clone();
            let file_state = file_outer.clone();
            if file_state.is_none() {
                return;
            }
            let f = (*file_state).clone().unwrap();
            loading.set(true);
            error_state.set(None);
            result_state.set(None);
            spawn_local(async move {
                let form = FormData::new().unwrap();
                form.append_with_blob_and_filename("image", &f, &f.name())
                    .ok();
                let init = web_sys::RequestInit::new();
                init.set_method("POST");
                init.set_body(&form.into());
                let url = format!("{}/predict", api_base());
                let request = web_sys::Request::new_with_str_and_init(&url, &init).unwrap();
                let win = window().unwrap();
                match wasm_bindgen_futures::JsFuture::from(win.fetch_with_request(&request)).await {
                    Ok(resp) => {
                        let resp: web_sys::Response = resp.dyn_into().unwrap();
                        if !resp.ok() {
                            let text = wasm_bindgen_futures::JsFuture::from(resp.text().unwrap())
                                .await
                                .ok()
                                .and_then(|t| t.as_string())
                                .unwrap_or_else(|| "Predict failed".to_string());
                            error_state.set(Some(text));
                        } else {
                            let text_js =
                                wasm_bindgen_futures::JsFuture::from(resp.text().unwrap()).await;
                            match text_js.map(|t| t.as_string().unwrap_or_default()) {
                                Ok(text) => match serde_json::from_str::<PredictResponse>(&text) {
                                    Ok(parsed) => result_state.set(Some(parsed)),
                                    Err(e) => error_state.set(Some(format!("Invalid JSON: {}", e))),
                                },
                                Err(_) => error_state.set(Some("Predict failed".into())),
                            }
                        }
                    }
                    Err(_) => error_state.set(Some("Network error".into())),
                }
                loading.set(false);
            });
        })
    };

    let server_label = match *server_ok {
        None => ("badge badge-wait", "checking..."),
        Some(true) => ("badge badge-ok", "online"),
        Some(false) => ("badge badge-warn", "offline"),
    };

    html! {
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <header class="flex flex-wrap items-center justify-between gap-4">
                <div>
                    <h1 class="m-0 font-extrabold text-2xl">{"Malaria Smear Analysis"}</h1>
                    <p class="opacity-80 text-sm mt-1">{"Upload a blood smear image and get a prediction from the model."}</p>
                </div>
                <div
                    class={classes!(
                        "badge", "text-white", "px-2", "py-0.5", "rounded", "text-xs",
                        match server_label.0 {
                            "badge badge-ok" => "bg-emerald-600",
                            "badge badge-warn" => "bg-red-600",
                            _ => "bg-white/30",
                        }
                    )}
                    title={format!("API base: {}", api_base())}
                >{format!("API {}", server_label.1)}</div>
            </header>

            <div class="bg-white/5 border border-white/10 rounded-xl p-6 mt-4" ondrop={on_drop} ondragover={on_drag_over}>
                <div class="flex flex-wrap gap-4">
                    <div class="flex-1 min-w-[280px]">
                        <label class="block text-sm mb-2 opacity-80">{"Choose image"}</label>
                        <input class="block" type="file" accept="image/*" onchange={on_file_change} />
                        <p class="text-xs opacity-60 mt-2">{"Or drag & drop an image onto this panel."}</p>
                    </div>
                    <div class="flex-1 min-w-[280px]">
                        if let Some(url) = (*preview_url).clone() {
                            <img class="max-h-80 w-full object-contain border border-white/10 rounded-lg" src={url} alt="preview" />
                        } else {
                            <div class="border border-dashed border-white/30 rounded-xl h-48 flex items-center justify-center opacity-80">
                                <span class="text-sm opacity-80">{"No image selected"}</span>
                            </div>
                        }
                    </div>
                </div>
                <div class="mt-4">
                    <button class={classes!(
                        "bg-emerald-600", "hover:bg-emerald-500", "text-white", "px-4", "py-2", "rounded-md",
                        if file.is_none() || *loading || *server_ok == Some(false) { Some("opacity-60") } else { None },
                        if file.is_none() || *loading || *server_ok == Some(false) { Some("cursor-not-allowed") } else { None }
                    )}
                        onclick={on_analyze.clone()} disabled={file.is_none() || *loading || *server_ok == Some(false)}>
                        { if *loading { "Analyzing..." } else { "Analyze" } }
                    </button>
                    if *server_ok == Some(false) {
                        <span class="ml-3 text-xs text-red-200">{"Server is offline. Start the Rust API at http://localhost:8080"}</span>
                    }
                </div>
            </div>

            <div class="bg-white/5 border border-white/10 rounded-xl p-6 mt-4">
                <h2 class="text-lg font-semibold mb-3">{"Result"}</h2>
                if result.is_none() && error.is_none() {
                    <p class="text-sm opacity-80">{"No result yet."}</p>
                }
                if let Some(err) = (*error).clone() {
                    <p class="text-sm text-red-300">{err}</p>
                }
                {
                    if let Some(res) = (*result).clone() {
                        let p_uninfected = *res.species_probabilities.get(4).unwrap_or(&0.0);
                        let p_infected = (1.0 - p_uninfected).clamp(0.0, 1.0);

                        let malaria_species_probs = [
                            *res.species_probabilities.first().unwrap_or(&0.0),
                            *res.species_probabilities.get(1).unwrap_or(&0.0),
                            *res.species_probabilities.get(2).unwrap_or(&0.0),
                            *res.species_probabilities.get(3).unwrap_or(&0.0),
                        ];
                        let malaria_species_labels = ["Falciparum", "Malariae", "Ovale", "Vivax"];
                        let best_species_idx = argmax(&malaria_species_probs);
                        let best_species_label = malaria_species_labels
                            .get(best_species_idx)
                            .unwrap_or(&"Unknown");
                        let best_species_conf = malaria_species_probs[best_species_idx];

                        let stage_labels = [
                            "Ring stage (R)",
                            "Trophozoite stage (T)",
                            "Schizont stage (S)",
                            "Gametocyte stage (G)",
                        ];
                        let best_stage_idx = argmax(&res.stage_probabilities);
                        let best_stage_label = stage_labels
                            .get(best_stage_idx)
                            .unwrap_or(&"Unknown stage");
                        let best_stage_conf = res.stage_probabilities[best_stage_idx];

                        html! {
                            <div>
                        <div class={classes!(
                            "rounded-xl",
                            "border",
                            "p-4",
                            "bg-black/20",
                            if res.infected { "border-red-500/40" } else { "border-emerald-500/40" }
                        )}>
                            <div class="flex items-center justify-between">
                                <div class="text-xs uppercase tracking-wide opacity-70">{"Diagnosis"}</div>
                                <button
                                    onclick={
                                        let res = res.clone();
                                        let preview_url = (*preview_url).clone();
                                        Callback::from(move |_| {
                                            let malaria_species_labels = ["Falciparum", "Malariae", "Ovale", "Vivax", "Uninfected"];
                                            let stage_labels = ["Ring stage (R)", "Trophozoite stage (T)", "Schizont stage (S)", "Gametocyte stage (G)"];

                                            let best_species_idx = argmax(&res.species_probabilities);
                                            let best_species_label = malaria_species_labels.get(best_species_idx).unwrap_or(&"Unknown");
                                            let best_species_conf = res.species_probabilities[best_species_idx];

                                            let best_stage_idx = argmax(&res.stage_probabilities);
                                            let best_stage_label = stage_labels.get(best_stage_idx).unwrap_or(&"Unknown");
                                            let best_stage_conf = res.stage_probabilities[best_stage_idx];

                                            let date = js_sys::Date::new_0().to_locale_string("en-US", &JsValue::from_str("{\"dateStyle\":\"long\", \"timeStyle\":\"short\"}")).as_string().unwrap_or_default();

                                            let data = serde_json::json!({
                                                "infected": res.infected,
                                                "species": best_species_label,
                                                "speciesProb": best_species_conf,
                                                "stage": best_stage_label,
                                                "stageProb": best_stage_conf,
                                                "date": date,
                                                "imageUrl": preview_url
                                            });

                                            if let Ok(js_data) = serde_wasm_bindgen::to_value(&data) {
                                                let window = web_sys::window().unwrap();
                                                let _ = js_sys::Reflect::get(&window, &JsValue::from_str("generateMalariaReport"))
                                                    .unwrap()
                                                    .dyn_into::<js_sys::Function>()
                                                    .unwrap()
                                                    .call1(&JsValue::NULL, &js_data);
                                            }
                                        })
                                    }
                                    class="text-[10px] bg-white/10 hover:bg-white/20 px-2 py-1 rounded transition-colors flex items-center gap-1"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                                    {"Download Report (PDF)"}
                                </button>
                            </div>
                            <div class="mt-1 text-xl font-extrabold">
                                { if res.infected {
                                    "Malaria detected"
                                } else {
                                    "No malaria detected"
                                }}
                            </div>
                            <div class="mt-2 text-sm opacity-90">
                                { if res.infected {
                                    html!{
                                        <span>{format!("Confidence: {}% (infected)", pct(p_infected) )}</span>
                                    }
                                } else {
                                    html!{
                                        <span>{format!("Confidence: {}% (uninfected)", pct(p_uninfected) )}</span>
                                    }
                                }}
                            </div>

                            <div class="mt-4 grid gap-3">
                                <div class="rounded-lg border border-white/10 bg-white/5 p-3">
                                    <div class="text-xs opacity-70">{"What this means"}</div>
                                    { if res.infected {
                                        html!{
                                            <div class="text-sm mt-1">
                                                {"The image shows features consistent with malaria infection."}
                                            </div>
                                        }
                                    } else {
                                        html!{
                                            <div class="text-sm mt-1">
                                                {"The image looks like a normal (uninfected) blood cell."}
                                            </div>
                                        }
                                    }}
                                </div>

                                { if res.infected {
                                    html!{
                                        <div class="rounded-lg border border-white/10 bg-white/5 p-3">
                                            <div class="text-xs opacity-70">{"Most likely malaria type"}</div>
                                            <div class="text-base font-semibold mt-1">{best_species_label}</div>
                                            <div class="text-sm opacity-90">{format!("Confidence: {}%", pct(best_species_conf))}</div>
                                        </div>
                                    }
                                } else {
                                    html!{}
                                }}

                                { if res.infected {
                                    html!{
                                        <div class="rounded-lg border border-white/10 bg-white/5 p-3">
                                            <div class="text-xs opacity-70">{"Most likely parasite stage (presence)"}</div>
                                            <div class="text-base font-semibold mt-1">{best_stage_label}</div>
                                            <div class="text-sm opacity-90">{format!("Confidence: {}%", pct(best_stage_conf))}</div>
                                            <div class="text-xs opacity-60 mt-1">{"Stage prediction is a weak label (image-level presence), not a per-parasite diagnosis."}</div>
                                        </div>
                                    }
                                } else {
                                    html!{}
                                }}
                            </div>
                        </div>

                        <details class="mt-4">
                            <summary class="cursor-pointer text-sm opacity-80">{"Show detailed probabilities"}</summary>
                            <div class="mt-3">
                                <div class="text-xs opacity-70 mb-1">{"Species probabilities"}</div>
                                <ProbBar label="Falciparum" value={*res.species_probabilities.first().unwrap_or(&0.0)} class_name="bar-a" />
                                <div class="h-2" />
                                <ProbBar label="Malariae" value={*res.species_probabilities.get(1).unwrap_or(&0.0)} class_name="bar-a" />
                                <div class="h-2" />
                                <ProbBar label="Ovale" value={*res.species_probabilities.get(2).unwrap_or(&0.0)} class_name="bar-a" />
                                <div class="h-2" />
                                <ProbBar label="Vivax" value={*res.species_probabilities.get(3).unwrap_or(&0.0)} class_name="bar-a" />
                                <div class="h-2" />
                                <ProbBar label="Uninfected" value={*res.species_probabilities.get(4).unwrap_or(&0.0)} class_name="bar-b" />
                            </div>

                            { if res.infected {
                                html!{
                                    <div class="mt-5">
                                        <div class="text-xs opacity-70 mb-1">{"Stage probabilities (presence)"}</div>
                                        <ProbBar label="Ring (R)" value={res.stage_probabilities[0]} class_name="bar-a" />
                                        <div class="h-2" />
                                        <ProbBar label="Trophozoite (T)" value={res.stage_probabilities[1]} class_name="bar-a" />
                                        <div class="h-2" />
                                        <ProbBar label="Schizont (S)" value={res.stage_probabilities[2]} class_name="bar-a" />
                                        <div class="h-2" />
                                        <ProbBar label="Gametocyte (G)" value={res.stage_probabilities[3]} class_name="bar-a" />
                                    </div>
                                }
                            } else {
                                html!{}
                            }}
                        </details>
                    </div>
                        }
                    } else {
                        html! {}
                    }
                }
            </div>
        </div>
    }
}

#[derive(Properties, PartialEq)]
struct ProbBarProps {
    label: AttrValue,
    value: f64,
    #[prop_or_default]
    class_name: AttrValue,
}

#[function_component(ProbBar)]
fn prob_bar(props: &ProbBarProps) -> Html {
    let pct = (props.value * 100.0).round().clamp(0.0, 100.0) as i32;
    html! {
        <div>
            <div class="flex justify-between text-[12px] opacity-75">
                <span>{props.label.clone()}</span>
                <span>{format!("{}%", pct)}</span>
            </div>
            <div class="h-2 bg-white/10 rounded-lg overflow-hidden">
                <div
                    class={classes!(
                        "h-2", "rounded-lg",
                        match props.class_name.as_ref() { "bar-b" => "bg-fuchsia-700", _ => "bg-sky-500" }
                    )}
                    style={format!("width:{}%", pct)}
                />
            </div>
        </div>
    }
}
