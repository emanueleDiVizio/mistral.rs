use mistralrs::{ModelDType, TextMessageRole, TextMessages, TextModelBuilder};
use mistralrs_stateful_runtime::{StatefulRequest, StatefulRuntime};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let runtime = StatefulRuntime::from_text_builder(
        TextModelBuilder::new("hf-internal-testing/tiny-random-LlamaForCausalLM")
            .with_dtype(ModelDType::F32)
            .with_force_cpu(),
    )
    .await?;

    let request = StatefulRequest::new(
        TextMessages::new().add_message(TextMessageRole::User, "Say hello in two words."),
    )
    .with_max_generated_tokens(8);

    let output = runtime.run_streaming(request).await?;
    for delta in output.deltas {
        if let Some(text) = delta.text_delta {
            print!("{text}");
        }
    }
    println!();
    Ok(())
}
