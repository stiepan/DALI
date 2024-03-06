from jax_playground.vit_dataloader import jaxline

jp = jaxline()
jp.build()

for _ in range(1000):
    jp.run()
