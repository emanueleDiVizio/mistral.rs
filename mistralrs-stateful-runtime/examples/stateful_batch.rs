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

    let requests = vec![
        StatefulRequest::new(
            TextMessages::new().add_message(TextMessageRole::User, "Say hello."),
        )
        .with_max_generated_tokens(8),
        StatefulRequest::new(
            TextMessages::new().add_message(TextMessageRole::User, "Say goodbye."),
        )
        .with_max_generated_tokens(8),
    ];

    let outputs = runtime.run_batch(requests).await?;
    println!(
        "stateful_batch_ok requests={} completion_tokens={}",
        outputs.len(),
        outputs
            .iter()
            .map(|out| out.stats.completion_tokens)
            .sum::<usize>()
    );
    Ok(())
}
