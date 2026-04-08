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
        TextMessages::new().add_message(TextMessageRole::User, "Count to three."),
    )
    .with_max_generated_tokens(8);

    let output = runtime.run(request).await?;
    println!(
        "stateful_single_seq_ok prompt_tokens={} completion_tokens={} finish_reason={:?} text={:?}",
        output.stats.prompt_tokens,
        output.stats.completion_tokens,
        output.finish_reason,
        output.text
    );
    Ok(())
}

