a=0
for i in *.jpg; do
  new=$(printf "%d.jpg" "$a") # %04d pad to length of 4
  mv -i -- "$i" "$new"
  let a=a+1
done