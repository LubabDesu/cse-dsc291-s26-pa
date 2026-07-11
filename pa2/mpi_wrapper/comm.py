from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.
        Your implementation should respect the passed reduction operator.
        The required operators for this assignment are MPI.MIN, MPI.SUM,
        and MPI.MAX.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        root = 0

        if rank == root : 
            dest_array[:] = src_array[:]
            for src_rank in range(1, nprocs) : 
                tmp = np.empty_like(src_array)
                self.comm.Recv(tmp, source=src_rank)
                if op == MPI.MIN : 
                    np.minimum(dest_array, tmp, out=dest_array)
                elif op == MPI.SUM : 
                    dest_array[:] += tmp
                elif op == MPI.MAX : 
                    np.maximum(dest_array, tmp, out=dest_array)
                else:
                    raise ValueError("Unsupported reduction op")
            
            for dst_rank in range(1, nprocs) : 
                self.comm.Send(dest_array, dest=dst_rank)
        else :
            self.comm.Send(src_array, dest=root)
            self.comm.Recv(dest_array, source=root)
 

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        assert src_array.size % nprocs == 0
        assert dest_array.size % nprocs == 0

        send_count = src_array.size // nprocs
        recv_count = dest_array.size // nprocs
        assert send_count == recv_count

        for peer in range(nprocs):
            send_start = peer * send_count
            send_end = send_start + send_count
            recv_start = peer * recv_count
            recv_end = recv_start + recv_count

            if peer == rank:
                dest_array[recv_start:recv_end] = src_array[send_start:send_end]
            else:
                self.comm.Sendrecv(
                    src_array[send_start:send_end],
                    dest=peer,
                    recvbuf=dest_array[recv_start:recv_end],
                    source=peer,
                )
