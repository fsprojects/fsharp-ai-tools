#!/bin/bash                                                                                   
echo "Converting fsx to inpynb"
for f in examples/*.fsx
do
        dotnet fsi tools/fsx2nb.fsx -i $f
done

echo "Executing inpynb with --allow-errors"
for f in examples/*.fsx
do
        jupyter nbconvert --to notebook --allow-errors --execute "${f%fsx}ipynb"
done

echo "Executing inpynb without --allow-errors"
for f in examples/*.fsx
do
        jupyter nbconvert --to notebook --execute "${f%fsx}ipynb"
done
