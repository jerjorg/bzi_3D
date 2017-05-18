  !!<summary>Find the Hermite normal form of a a given integer
  !!matrix. Similar to the SNF finder above but a little simpler. This
  !!routine is not very elegant, just brute force. Don't be
  !!disappointed.</summary>
  !!<parameter name="S" regular="true">The 3x3 integer matrix
  !!describing the relationship between two commensurate
  !!lattices.</parameter>
  !!<parameter name="H" regular="true">The resulting HNF
  !!matrix.</parameter>
  !!<parameter name="B" regular="true">The transformation matrix such
  !!that $H = SB$.</parameter>
  subroutine HermiteNormalForm(S,H,B)
    integer, intent(in) :: S(3,3) 
    integer, intent(out), dimension(3,3) :: H, B 

    !!<local name="tempcol">When two columns need to be swapped,
    !!tempcol mediates the swap.</local>
    !!<local name="check">Stores the HNF as a vector; only used as
    !!failsafe for meeting HNF criteria.</local>
    !!<local name="maxidx, minidx">The indices of max/min values in a
    !!column.</local>
    !!<local name="multiple">When reducing the columns to HNF form,
    !!the multiplicative relationship between two elements of two
    !!columns; reused.</local>
    !!<local name="minm">The smallest element in row 1 of the HNF
    !!being constructed.</local>

    integer :: i, minm, maxidx, minidx, multiple, j,  check(9), tempcol(3)

    if (determinant(S) == 0) stop "Singular matrix passed to HNF routine"
    B = 0; H = S ! H starts out as S, the input matrix
    forall(i=1:3); B(i,i) = 1; end forall ! B = identity

    do ! Keep doing column operations until all elements in row 1 are
       ! zero except the one on the diagonal. 
       ! Divide the column with the smallest value into the largest
       do while (count(H(1,:)/=0) > 1) ! Keep going until only zeros beyond first element
          call get_minmax_indices(H(1,:),minidx,maxidx)
          minm = H(1,minidx)
          ! Subtract a multiple of the column containing the smallest element from
          ! the row containing the largest element
          multiple = H(1,maxidx)/minm ! Factor to multiply by
          H(:,maxidx) = H(:,maxidx) - multiple*H(:,minidx)
          B(:,maxidx) = B(:,maxidx) - multiple*B(:,minidx)
          if (any(matmul(S,B)/=H)) stop "COLS: Transformation matrices didn't work"
       enddo ! End of step row 1
       if (H(1,1) == 0) call swap_column(H,B,1) ! swap columns if (1,1) is zero
       if (H(1,1)<0)then; H(:,1) = -H(:,1); B(:,1) = -B(:,1) ! Change sign if (1,1) is negative
       endif
       if (count(H(1,:)/=0) > 1) stop "Didn't zero out the rest of the row"
       if (any(matmul(S,B)/=H)) stop "COLSWAP: Transformation matrices didn't work"
       exit
    enddo
    ! Now work on element H(2,3)
    do
       do while (H(2,3)/=0)
          if (H(2,2) == 0) then
             tempcol = H(:,2); H(:,2) = H(:,3); H(:,3) = tempcol 
             tempcol = B(:,2); B(:,2) = B(:,3); B(:,3) = tempcol 
             if (H(2,3) == 0) exit
          endif
          if (abs(H(2,3))<abs(H(2,2))) then; maxidx = 2; minidx = 3
          else; maxidx = 3; minidx = 2; endif
          multiple = H(2,maxidx)/H(2,minidx)
          H(:,maxidx) = H(:,maxidx) - multiple*H(:,minidx)
          B(:,maxidx) = B(:,maxidx) - multiple*B(:,minidx)
          if (any(matmul(S,B)/=H)) stop "COLS: Transformation matrices didn't work"
       enddo
       if (H(2,2) == 0) then
          tempcol = H(:,2); H(:,2) = H(:,3); H(:,3) = tempcol 
       endif
       if (H(2,2)<0) then ! Change signs
       H(:,2) = -H(:,2); B(:,2) = -B(:,2); endif
       if (H(2,3)/=0) stop "Didn't zero out last element"

       if (any(matmul(S,B)/=H)) stop "COLSWAP: Transformation matrices didn't work"
       exit
    enddo
    if (H(3,3)<0) then ! Change signs
       H(:,3) = -H(:,3); B(:,3) = -B(:,3);
    endif
    check = reshape(H,(/9/))
    if (any(check((/4,7,8/))/=0)) stop "Not lower triangular"
    if (any(matmul(S,B)/=H)) stop "END PART1: Transformation matrices didn't work"
 
    ! Now that the matrix is in lower triangular form, make sure the lower off-diagonal 
    ! elements are non-negative but less than the diagonal elements
    do while (H(2,2) <= H(2,1) .or. H(2,1)<0)
       if (H(2,2) <= H(2,1)) then; multiple = 1
       else; multiple = -1; endif
       H(:,1) = H(:,1) - multiple*H(:,2)
       B(:,1) = B(:,1) - multiple*B(:,2)
    enddo
    do j = 1,2
       do while (H(3,3) <= H(3,j) .or. H(3,j)<0)
          if (H(3,3) <= H(3,j)) then; multiple = 1
          else; multiple = -1; endif
          H(:,j) = H(:,j) - multiple*H(:,3)
          B(:,j) = B(:,j) - multiple*B(:,3)
       enddo
    enddo
    if (any(matmul(S,B)/=H)) stop "END: Transformation matrices didn't work"
    check = reshape(H,(/9/))
    if (any(check((/4,7,8/))/=0)) stop "Not lower triangular"
    if (any(check((/2,3,6/))<0)) stop "Negative elements in lower triangle"
    if (check(2) > check(5) .or. check(3) > check(9) .or. check(6) > check(9)) stop "Lower triangular elements bigger than diagonal"
  ENDSUBROUTINE HermiteNormalForm
