mkdir -p backup_caches
cp cache1.json backup_caches/cache1.json
cp cache2.json backup_caches/cache2.json
cp cache3.json backup_caches/cache3.json
cp cache4.json backup_caches/cache4.json
cp cache5.json backup_caches/cache5.json
cp cache6.json backup_caches/cache6.json

python merge_caches.py cache.json cache1.json cache2.json cache3.json cache4.json cache5.json cache6.json cache.json

cp cache.json cache1.json
cp cache.json cache2.json
cp cache.json cache3.json
cp cache.json cache4.json
cp cache.json cache5.json
cp cache.json cache6.json
