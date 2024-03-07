import time
from jax_playground.vit_dataloader import jaxline

jp = jaxline()
jp.build()

start = time.time()

for _ in range(1000):
# for _ in range(1):
    jp.run()

end = time.time()
print(end - start)
