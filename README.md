# NCKHSV2022
## Sử dụng ảnh MRI chụp u não để segmentation vùng u 
### Các bước làm 
- Đọc ảnh bảng MRI bằng opencv và đọc file annotation để vẽ các mask cho ảnh u não 
- Data augmentation để tăng cường số lượng ảnh 
- Data Visualization 
- Chuẩn hóa lại dữ liệu
- Sử dụng model Unet 
- Kết quả đạt 92% trên tập train và 81% trên tập validation 
- Deploy lên website với frontend: html , css , js , boostrap 4 , backend: flask , model bên trên
