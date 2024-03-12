import time
from jax_playground.vit_dataloader import jaxline, dali_pipeline

p = jaxline()
# p = dali_pipeline()
p.build()

start = time.time()

for _ in range(1000):
# for _ in range(1):
    p.run()

end = time.time()
print(end - start)
