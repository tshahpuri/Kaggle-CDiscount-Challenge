from tqdm import *

gen = contiguous_read(TRAIN_BSON_FILE)
for _ in tqdm(range(10000)):
    obs = next(gen)

gen = block_reader(TRAIN_BSON_FILE, meta_data, chunk_size=1000, shuffle=False)
next(gen) #warm up
for _ in tqdm(range(10000)):
    obs = next(gen)
    
gen = block_reader(TRAIN_BSON_FILE, meta_data.sample(100000), chunk_size=1000, shuffle=True)
next(gen) #warm up
for _ in tqdm(range(10000)):
    obs = next(gen)
    
gen = BSONIterator(TRAIN_BSON_FILE, batch_size=256)
next(gen) #warm up
for _ in tqdm(range(1000)):
    batch_data = next(gen)

train_meta = meta_data.sample(frac=0.9, replace=False)

gen = BSONIterator(TRAIN_BSON_FILE, batch_size=256, shuffle=True, metadata=train_meta)
next(gen) #warm up
for _ in tqdm(range(1000)):
    batch_data = next(gen)
    
import functools

gen = BSONIterator(TRAIN_BSON_FILE, batch_size=256, shuffle=True, metadata=train_meta, 
                   preprocess_batch_func=functools.partial(preprocess_batch, 
                                                           img_size=128, 
                                                           labels=meta_data.category_id))
next(gen) #warm up
for _ in tqdm(range(100)):
    batch_data = next(gen)