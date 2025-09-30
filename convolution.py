import streamlit as st
import numpy as np

# --- 1. HÀM HỖ TRỢ XỬ LÝ MA TRẬN VÀ TRUNG VỊ ---

def calculate_median(arr: list, mode: str) -> float:
    """Tính giá trị trung vị cho mảng. Hỗ trợ chế độ chẵn."""
    if not arr:
        return 0.0
    
    # Sắp xếp mảng để tìm trung vị
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    
    if n % 2 != 0:
        # Trường hợp Lẻ: Trả về phần tử chính giữa
        return sorted_arr[n // 2]
    else:
        # Trường hợp Chẵn: Có hai phần tử giữa
        val1 = sorted_arr[n // 2 - 1]
        val2 = sorted_arr[n // 2]
        
        if mode == 'first':
            # Theo yêu cầu: lấy giá trị đầu tiên trong hai giá trị giữa
            return val1
        elif mode == 'average':
            # Tính trung bình hai giá trị giữa
            return (val1 + val2) / 2.0
        return 0.0

def create_matrix_input(label: str, rows: int, cols: int, key_prefix: str) -> np.ndarray or None:
    """Tạo giao diện nhập ma trận dưới dạng cột và trả về mảng numpy."""
    st.subheader(label)
    
    # Tạo các cột cho input
    cols_input = st.columns(cols)
    matrix = []
    
    # Dữ liệu mặc định cho dễ thử nghiệm
    default_input = np.array([[1, 2, 1, 0], [5, 6, 2, 3], [4, 1, 9, 8], [0, 2, 3, 1]])
    default_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    
    # Chọn dữ liệu mặc định
    if key_prefix == 'input_':
        default_data = default_input
    elif key_prefix == 'kernel_':
        default_data = default_kernel
    else:
        default_data = np.zeros((rows, cols))

    # Xử lý ma trận
    try:
        for r in range(rows):
            row = []
            for c in range(cols):
                
                # Lấy giá trị mặc định an toàn
                default_val = default_data[r, c] if r < default_data.shape[0] and c < default_data.shape[1] else 0.0
                
                # Tạo widget input trong từng cột
                val = cols_input[c].number_input(
                    label=f'{key_prefix}{r},{c}',
                    value=float(default_val),
                    key=f'{key_prefix}{r}_{c}',
                    label_visibility='collapsed',
                    step=1.0 # Bước nhảy là 1.0 cho số nguyên/float
                )
                row.append(val)
            matrix.append(row)
        return np.array(matrix)
    except Exception as e:
        st.error(f"Lỗi khi đọc ma trận {label}: {e}")
        return None

# --- 2. HÀM TÍNH TOÁN TÍCH CHẬP ---

def perform_convolution(input_matrix, kernel, P, S):
    """Thực hiện phép tích chập (Convolution) với Padding và Stride."""
    R, C = input_matrix.shape
    KR, KC = kernel.shape
    
    # 1. Tính kích thước đầu ra
    OR = np.floor((R - KR + 2 * P) / S).astype(int) + 1
    OC = np.floor((C - KC + 2 * P) / S).astype(int) + 1

    if OR <= 0 or OC <= 0:
        raise ValueError(f"Kích thước ma trận đầu ra không hợp lệ: {OR}x{OC}. Vui lòng kiểm tra lại tham số Kernel/Padding/Stride.")

    # 2. Thêm Padding (Zero Padding)
    padded_matrix = np.pad(input_matrix, ((P, P), (P, P)), 'constant', constant_values=0)
    
    # 3. Thực hiện Tích chập
    output_matrix = np.zeros((OR, OC))
    
    for i in range(OR):
        for j in range(OC):
            # Xác định vùng Kernel trên ma trận đệm
            start_row = i * S
            end_row = start_row + KR
            start_col = j * S
            end_col = start_col + KC
            
            # Trích xuất cửa sổ và tính tổng tích
            window = padded_matrix[start_row:end_row, start_col:end_col]
            output_matrix[i, j] = np.sum(window * kernel)
            
    return output_matrix

# --- 3. HÀM TÍNH TOÁN LỌC TRUNG VỊ ---

def perform_median_filter(input_matrix, N, mode):
    """Thực hiện Lọc Trung vị (Median Filter)"""
    R, C = input_matrix.shape
    
    # Kích thước bán kính padding (window radius)
    P_median = N // 2
    
    output_matrix = np.zeros((R, C))
    
    # 1. Thêm Zero Padding cho ma trận đầu vào (tương đương với việc kiểm tra biên)
    padded_input = np.pad(input_matrix, P_median, 'constant', constant_values=0)
    
    for r in range(R):
        for c in range(C):
            neighbors = []
            
            # Lặp qua cửa sổ N x N (lấy từ ma trận đã đệm)
            for i in range(N):
                for j in range(N):
                    # Vị trí trên ma trận đã đệm
                    pr = r + i
                    pc = c + j
                    neighbors.append(padded_input[pr, pc])

            # Tính trung vị và gán vào ma trận đầu ra
            output_matrix[r, c] = calculate_median(neighbors, mode)
            
    return output_matrix


# --- 4. GIAO DIỆN STREAMLIT CHÍNH ---

def main():
    st.set_page_config(layout="centered", page_title="Công cụ Tích chập & Lọc Trung vị")
    st.title("🔢 Công cụ Tích chập và Lọc Trung vị")
    
    # --- PHẦN 1: NHẬP LIỆU CHO TÍCH CHẬP ---
    st.header("1. Tham số Tích chập (Convolution)")

    # 1.1 Kích thước Ma trận Đầu vào
    col1, col2 = st.columns(2)
    R = col1.number_input("Số hàng Ma trận Đầu vào (R)", min_value=1, value=4, key='R')
    C = col2.number_input("Số cột Ma trận Đầu vào (C)", min_value=1, value=4, key='C')
    
    input_matrix = create_matrix_input("Ma trận Đầu vào (Input)", R, C, 'input_')

    # 1.2 Kích thước Kernel
    col3, col4 = st.columns(2)
    KR = col3.number_input("Số hàng Kernel (KR)", min_value=1, value=3, key='KR')
    KC = col4.number_input("Số cột Kernel (KC)", min_value=1, value=3, key='KC')
    
    kernel_matrix = create_matrix_input("Ma trận Kernel/Bộ lọc", KR, KC, 'kernel_')
    
    # 1.3 Tham số Padding và Stride
    st.markdown("---")
    st.subheader("Tham số Khác")
    col5, col6 = st.columns(2)
    P = col5.number_input("Padding (P)", min_value=0, value=0, key='P', step=1)
    S = col6.number_input("Stride (S)", min_value=1, value=1, key='S', step=1)

    # Nút Thực hiện Tích chập
    if st.button("▶️ Thực hiện Tích chập", key='btn_conv'):
        if input_matrix is None or kernel_matrix is None:
            st.error("Vui lòng kiểm tra lại kích thước và giá trị ma trận.")
        else:
            try:
                # Lưu kết quả vào state để dùng cho Lọc Trung vị
                st.session_state['conv_result'] = perform_convolution(input_matrix, kernel_matrix, P, S)
                st.success("Tích chập hoàn tất!")
            except ValueError as e:
                st.error(f"Lỗi tính toán Tích chập: {e}")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi không xác định: {e}")

    # --- PHẦN 2: KẾT QUẢ TÍCH CHẬP ---
    st.header("2. Kết quả Ma trận Tích chập")
    
    if 'conv_result' in st.session_state and st.session_state['conv_result'].size > 0:
        conv_matrix = st.session_state['conv_result']
        OR, OC = conv_matrix.shape
        st.info(f"Kích thước Đầu ra (Feature Map): **{OR} x {OC}**")
        
        # Hiển thị ma trận kết quả
        st.dataframe(conv_matrix.round(2), use_container_width=True)
        
        # --- PHẦN 3: LỌC TRUNG VỊ ---
        st.header("3. Lọc Trung vị (Median Filtering)")
        
        col7, col8 = st.columns(2)
        N = col7.number_input(
            "Kích thước Window (N x N)", 
            min_value=1, 
            value=3, 
            step=2, 
            key='N',
            help="Kích thước Window/Neighbor. Thường là số lẻ (ví dụ: 3, 5)."
        )
        
        median_mode = col8.selectbox(
            "Chế độ Trung vị Chẵn",
            options=['first', 'average'],
            format_func=lambda x: 'Chọn giá trị đầu tiên (theo yêu cầu)' if x == 'first' else 'Tính trung bình hai giá trị',
            key='median_mode'
        )
        
        if st.button("✨ Thực hiện Lọc Trung vị", key='btn_median'):
            if N % 2 == 0:
                st.error("Kích thước Window (N) phải là số lẻ.")
            else:
                try:
                    median_result = perform_median_filter(conv_matrix, N, median_mode)
                    st.session_state['median_result'] = median_result
                    st.success("Lọc Trung vị hoàn tất!")
                except Exception as e:
                    st.error(f"Lỗi tính toán Lọc Trung vị: {e}")

    # --- KẾT QUẢ LỌC TRUNG VỊ ---
    if 'median_result' in st.session_state and st.session_state['median_result'].size > 0:
        st.subheader("Kết quả Lọc Trung vị")
        st.dataframe(st.session_state['median_result'].round(2), use_container_width=True)
    
if __name__ == "__main__":
    main()
