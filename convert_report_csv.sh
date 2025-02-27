# nsys stats --report cuda_kern_exec_trace --format csv,column --output .,- report/v100/v100_*.nsys-rep 
# nsys stats --report cuda_gpu_mem_time_sum --format csv,column --output .,- v100_t_16_img_256.nsys-rep 

for file in report/v100/v100_*.nsys-rep; do
    filename=$(basename "$file" .nsys-rep)
    
    # Kernel execution trace
    nsys stats --report cuda_kern_exec_trace --format csv,column \
        --output "report/v100/${filename}_kern_exec_trace.csv" "$file"

    # GPU memory usage summary
    nsys stats --report cuda_gpu_mem_time_sum --format csv,column \
        --output "report/v100/${filename}_gpu_mem_time_sum.csv" "$file"
done
