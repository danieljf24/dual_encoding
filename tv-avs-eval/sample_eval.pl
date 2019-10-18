#!/usr/bin/perl

# This code implements measures discussed in the SIGIR 2008 paper
# "A Simple and Efficient Sampling Method for Estimating AP and NDCG"
# by Emine Yilmaz, Evangelos Kanoulas, and Javed A. Aslam. See the ACM
# Digital Library or www.ccs.neu.edu/home/ekanou/research/papers/mypapers/sigir08b.pdf
# 
# The code implements the measures xinfAP and NDCG. xinfAP is an extension
# of infAP and allows for random sampling at different rates for different
# strata of the pooled system output to be judged. For this measure the
# ground truth (qrels) contain an extra field identifying which stratum
# each shot comes from.
#
# Recipients of this software assume all responsibilities associated with 
# its operation, modification and maintenance.
# 
# CHANGE LOG 
#
# 27 Aug 10; NIST modified output to be more like trec_eval's; added estimated
# number relevant retrieved, estimated number relevant, number retrieved.
#
# 24 Aug 10: NIST replaced constant "1000" (max result size for TREC) 
# with a variable $maxResultSize to accommodate TRECVID's max size 
# of 2000,etc.
#
# 7 Aug 10: Original code by Emine Yilmaz modifed to output inferred 11pt 
# interpolated precision/recall values
#
# 5 Sep 12: Code added to adjust average precision iff the total number
# of inferred relevant exceeds the maximum result set size. OKed by
# Emine Yilmaz  $AP = $AP * $num_rels{$topic}/$maxResultSize;

if (@ARGV < 2) {
  die "Usage:  sample_eval.pl [-q] <qrel_file> <trec_file>\n\n";
}

#print "\n\n\nARGV gelio : @ARGV";

# Get names of qrel and trec files; check for -q option.

if (@ARGV == 3) {
  shift;                                # Remove -q.
  $print_all_queries = 1;
  }

$qrel_file = shift;                     # Shift implicitly acts on @ARGV.
$trec_file = shift;

# look for -G option  with <lvl> = <num> where level is relevance level
# and num is the gain value
#
if ($#ARGV > 0)  {  # there is the -G option
  shift;  # -G option removed
  for($i=0;$i<=$#ARGV;$i++) {
     $rel_map = $ARGV[$i];
     # extract the relevance
     $original_rel = substr($rel_map, 0,1);
     $updated_rel = substr($rel_map, 2,3);
     $rel_mappings{$original_rel} = $updated_rel;
  }
}

$maxResultSize = 1000;  # Adjust as needed

@precisionranks = (10,100,1000,$maxResultSize);


for ($i=0;$i<=$#precisionranks;$i++){
    $meanprecs[$i]=0;
}

# Process qrel file first.

open(QREL, $qrel_file) or
  die "Failed to open $qrel_file: $!\n\n";

{
local $/ = undef;                       # Reads grab the whole file.
@data = split(/\s+/, <QREL>);           # Data array has all values from the
}                                       # file consecutively.

close(QREL) or
  die "Couldn't close $qrel_file: $!\n\n";


#Initialize the mean values
$sum_ndcg = 0;
$sum_avg_prec = 0;
$sum_rel_ret = 0;
$sum_num_rel = 0;
 
# Now take the values from the data array (four at a time) and
# put them in a data structure.  Here's how it will work.
#
# %qrel is a hash whose keys are topic IDs and whose values are
# references to hashes.  Each referenced hash has keys which are
# doc IDs and values which are relevance values.  In other words...
#
# %qrel                         The qrel hash.
# $qrel{$topic}                 Reference to a hash for $topic.
# $qrel{$topic}->{$doc_id}      The relevance of $doc_id in $topic.
# $category{$topic}->{$doc_id}      The category (which subpool) of $doc_id in $topic.

# Now the sampled values for each category
# $sampled_rel{$topic}->{$category}	number of sampled relevant documents within each $category
# $sampled_rels_per_grade{$topic}->{$category}->{$rel}	 number of sampled documents that have relevance grade $rel
# $sampled_docs{$topic}->{$category}	number of sampled documents within each category
# $docs_per_category{$topic}->{$category}	 	number of documents within each category		
# $num_rel{$topic}               Hash whose values are (estimated) number
#                               of docs relevant for each topic.
# $num_rels_per_grade{$topic}->{$rel}		estimated number of documents with relevance grade $rel

 while (($topic, $dummy, $doc_id, $doc_category, $rel) = splice(@data,0,5)) {
  if(exists($rel_mappings{$rel})) {  # if a relevance mapping is provided, map it
     $mapped_rel = $rel_mappings{$rel};
     $rel = $mapped_rel;
  }  

  $qrel{$topic}->{$doc_id} = $rel;
  $category{$topic}->{$doc_id} = $doc_category; 
  $num_rel{$topic} += $rel;
  $docs_per_category{$topic}->{$doc_category} += 1;
  if($rel >= 0)
  {
    $sampled_docs{$topic}->{$doc_category} += 1;
  }    
  if($rel > 0)
  {
    $sampled_rels_per_grade{$topic}->{$doc_category}->{$rel} += 1;
    $sampled_rel{$topic}->{$doc_category} += 1;
  } 
} #end while

# Estimate the total number of relevant documents for each topic (needed by AP)
foreach $topic (sort keys %qrel) {
    foreach $doc_category (sort keys %{$docs_per_category{$topic}}) 
    {
	if($sampled_docs{$topic}->{$doc_category} !=0) {
	    $rel_estimates_category = $sampled_rel{$topic}->{$doc_category}*$docs_per_category{$topic}->{$doc_category}/$sampled_docs{$topic}->{$doc_category};
	    $num_rels{$topic} += $rel_estimates_category;
	} #end if
    } #end foreach

    $sum_num_rel += $num_rels{$topic}; #sum of estimated relevant across topics

} #end foreach

# Estimate the optimal DCG value (discount function 1/log(r+1))
# To compute, first estimate the estimated number of relevant documents within each grade
foreach $topic (sort keys %qrel) {
  foreach $doc_category (sort keys %{$docs_per_category{$topic}}) {
    foreach $rel_grade (sort keys %{$sampled_rels_per_grade{$topic}->{$doc_category}}) { 
      $num_rels_per_grade{$topic}->{$rel_grade} += ($sampled_rels_per_grade{$topic}->{$doc_category}->{$rel_grade})*$docs_per_category{$topic}->{$doc_category}/$sampled_docs{$topic}->{$doc_category};
    }
  }
}

# Now you can compute the optimal dcg value
foreach $topic (sort keys %qrel) {
  $start_rank = 0;
  foreach $rel_grade (reverse sort keys %{$num_rels_per_grade{$topic}})  {
    for ($r=($start_rank+1);$r<=($start_rank+$num_rels_per_grade{$topic}->{$rel_grade});$r++)  {
       $optimal_dcg{$topic} += $rel_grade/(log($r+1)/log(2));
       if($r>=$maxResultSize) {  # systems are not allowed to retrieve more than maxResultSetSize docs
          last; }
      }
    $start_rank += $num_rels_per_grade{$topic}->{$rel_grade}; 
  } 
 }#

# prints estimated number of relevants
# foreach $topic (sort keys %qrel) {
# $num_rel_docs = $num_rels{$topic};
# print "$topic $num_rel_docs\n";
#}

# Now process the trec file.

open(TREC, $trec_file) or
  die "Failed to open $trec_file: $!\n\n";

{
local $/ = undef;                       # Reads grab the whole file.
@data = split(/\s+/, <TREC>);           # Data array has all values from the
}                                       # file consecutively.

close(TREC) or
  die "Couldn't close $qrel_file: $!\n\n";

# Process the trec_file data in much the same manner as above.

%num_ret = ();    # Initialize hash to hold number of items returned by topic 
$sum_num_ret = 0; # Initialize sum of items returned for all topics
while (($topic, $dummy, $doc_id, $dummy, $score, $dummy) = splice(@data,0,6)) {
  #$topic =~ s/^0*//;
  $trec{$topic}->{$doc_id} = $score;
  $num_ret{$topic}++;
  #$sum_num_ret++;
 }


foreach $topic (sort {$a <=> $b} keys %trec) {  # Process topics in order.
  next unless exists $qrel{$topic};
  $num_topics++;                        # Processing another topic...
  $href = $trec{$topic};                # Get hash pointer.

  # Now sort doc IDs based on scores and calculate stats.
  # Note:  Break score ties lexicographically based on doc IDs.
  # Note2: Explicitly quit after $maxResultSize docs to conform to TREC while still
  #        handling trec_files with possibly more docs.

  # SAP_category{$category}	# holds the sum of the precisions at relevant document wihtin each category
  # gain_category{$category}    # discounted gain values within each category
  # $num_sampled{$category}	# number of sampled documents within $category upto current rank
  # $num_relevant{$category}	# number of sampled relevant documents within $category upto current rank
  # $num_docs{#category}	# number of documents that fall in to $category upto current rank
  # $num_depth100		# number of depth100 documents upto current rank

 $num_depth100 = 0;
 $rank = 0;

 $num_ret = 0;                         # Initialize number retrieved.
 $num_rel_ret = 0;                     # Initialize number relevant retrieved.
 $sum_prec = 0;                        # Initialize sum precision.

 # Initialize the hashes 
 %SAP_category = ();
 %gain_category = ();
 %num_sampled = ();
 %num_relevant = ();
 %num_docs = ();

 foreach $doc_id (sort
    { ($href->{$b} <=> $href->{$a}) || ($b cmp $a) } keys %$href) {
    $rank = $rank +1; 
    $sum_num_ret++;
   
      $rel = $qrel{$topic}->{$doc_id};    # Doc's relevance.
      $doc_category = $category{$topic}->{$doc_id}; # The category of this document
      
      if ($rel > 0) { # this document is relevant
	  # estimate the precision above this relevant document
	  $prec_above = 0;
	  foreach $category_val (sort keys %{$docs_per_category{$topic}}){
	      # compute precisions for all categories		           
	      if($num_depth100!= 0) {
		  # probability of picking a document from this category
		  $prob_category = $num_docs{$category_val}/$num_depth100;
		  if($prob_category !=0) {
		      $prec_above += $prob_category*($num_relevant{$category_val} + 0.00001)/($num_sampled{$category_val} + 0.00003);
		  }
	      }
	  }
	  # estimated precision at relevant document
	  $prec = 1/$rank + ($num_depth100/$rank)*$prec_above; 
	  $SAP_category{$doc_category} += $prec;     
	  $num_relevant{$doc_category} += 1;
	  
	  # compute the discounted cumulative gain within this category
	  $gain_category{$doc_category} += $rel/(log($rank+1)/log(2));
      }

      #print "HERE IS ANOTHER LOOP\n";
      
      if(exists($qrel{$topic}->{$doc_id})) { # this document is in depth 100 pool
	  $num_depth100 += 1; 
	  $num_docs{$doc_category} += 1;

	  if ($rel >= 0) { # this document is sampled
		  #print "UPDATING\n\n\n";
	      $num_sampled{$doc_category} += 1;
	  }
      }
      
      
      # Estimate number of relevant documents at each rank
      $num_rel_rank_k=0;
      foreach $category_val (sort keys %{$docs_per_category{$topic}}){
	  $num_rel_rank_k{$topic}{$rank} += $num_docs{$category_val}*($num_relevant{$category_val} + 0.00001)/($num_sampled{$category_val} + 0.00003);
      }
      $num_rel_ret{$topic}=$num_rel_rank_k{$topic}{$rank};
      
      foreach $cutoff (@precisionranks)
      {
	      if($rank == $cutoff)
	      {
		      $precision{$topic}->{$rank} = $num_rel_rank_k{$topic}->{$rank}/$cutoff;

	      }

      }

      if ($rank >= $maxResultSize) {
	  last;
      }

  }

  # Now fill in the rest of the precision values
  foreach $cutoff (@precisionranks)
  {
	  if (not exists($precision{$topic}->{$cutoff}))
	  {
		  $precision{$topic}->{$cutoff} = $num_rel_ret{$topic}/$cutoff;
	  }
  }

  # Now estimate the average precision value
  $AP = 0;
  foreach $category_val (sort keys %{$docs_per_category{$topic}}){
      if($sampled_docs{$topic}->{$category_val} !=0) {
	  
	  #estimated number of relevant documents that fall in this category
	  $rel_estimates_category = $sampled_rel{$topic}->{$category_val}*$docs_per_category{$topic}->{$category_val}/$sampled_docs{$topic}->{$category_val};
        
          if($num_rels{$topic} != 0) {
          # probability fo picking a relevant document from this category
          $prob_category = $rel_estimates_category/$num_rels{$topic};
         
          # expected value of average precision within this category
          $AP_category = 0;
          if($sampled_rel{$topic}->{$category_val} != 0) {
          $AP_category = $SAP_category{$category_val}/$sampled_rel{$topic}->{$category_val};
         } 
          # expected value of average precision
          $AP += $prob_category*$AP_category;
        } # end if
     } #end if
  } # end foreach

  # PO 15. August 2012
  # Adjust AP in case inferred number relevant is greater than result set size
  if ($num_rels{$topic} > $maxResultSize)
  {
        $AP = $AP * $num_rels{$topic}/$maxResultSize; 
  }

  # estimate the dcg value
   $dcg_val = 0;
   foreach $category_val (sort keys %{$docs_per_category{$topic}}){
     if($num_depth100!= 0) {
        # probability of picking a document from this category
        $prob_category = $num_docs{$category_val}/$num_depth100;
         
        if($num_sampled{$category_val} != 0) {
          $dcg_val += $prob_category*$gain_category{$category_val}/$num_sampled{$category_val};
        }
      }
   }

  #Now compute the NDCG value
  $ndcg_val = 0; 
  if($optimal_dcg{$topic} != 0) {
    $ndcg_val = $num_depth100*$dcg_val/$optimal_dcg{$topic};
   }

  #print "OPTIMAL DCG : $optimal_dcg{$topic} DEPTH100 : $num_depth100 DCG VAL : $dcg_val\n"; 
  #exit;

 if ($print_all_queries) {
         printf "infAP\t\t$topic\t\t%6.4f\n", $AP;
	 printf "infNDCG\t\t$topic\t\t%6.4f\n", $ndcg_val;
	 foreach $cutoff (@precisionranks)
	 {
		 printf "iP$cutoff\t\t$topic\t\t%6.4f\n",$precision{$topic}->{$cutoff};
	 }
	 printf "inum_rel_ret\t$topic\t%14.4f\n", $num_rel_ret{$topic};
	 printf "inum_rel\t$topic\t%14.4f\n", $num_rels{$topic};
	 printf "num_ret\t\t$topic\t%9d\n", $num_ret{$topic};
	 
  }

 $sum_avg_prec += $AP;
 $sum_ndcg += $ndcg_val;
 $sum_rel_ret += $num_rel_ret{$topic};

 
  for ($i=0;$i<=$#precisionranks;$i++){
    $cutoff = $precisionranks[$i];
    $meanprecs[$i] += $precision{$topic}->{$cutoff};
  }


}


# 07-Aug-2010 Added interpolated Precision averaged over 11 recall points
#      Average interpolated at the given recall points - default is the 11 points.
#      Both map, 11-pt_avg and R-prec can be regarded as estimates of the area under
#      the standard interpolated recall-precision (ircl_prn) curve.


@cutoff_array = ("0",".1",".2",".3",".4",".5",".6",".7",".8",".9","1"); #cut-off levels

$sum_int11ptAP=0;


for ($i=0;$i<=10;$i++){
    $int11ptP[$i]=0;
}

foreach $topic (sort keys %num_rel_rank_k) {
    @cutoffs = map { $_ * $num_rels{$topic}} @cutoff_array; #cut-offs expressed in estimated num of rel docs
    $int_prec = 0;
    $int11ptAP=0;

    $current_cut = $#cutoffs;

    while ($current_cut>=0 & $cutoffs[$current_cut]>$num_rel_ret{$topic}){
	$current_cut --;
    }


    #  Loop over all retrieved docs in reverse order.  Needs to be
    #  reverse order since we are calcualting interpolated precision.
    #  int_prec(r) defined to be max(prec(r')) for all r' >= r.

    foreach $rank (sort {$b <=> $a} keys %{$num_rel_rank_k{$topic}}){
	
	# Estimated interpolated precision
	$prec = $num_rel_rank_k{$topic}{$rank}/$rank;
	if ($int_prec < $prec){$int_prec=$prec;}
#	print "num retrieved $num_rel_rank_k{$topic}{$rank} $cutoffs[$current_cut]\n";
	if ($current_cut>=0 & $cutoffs[$current_cut]>$num_rel_rank_k{$topic}{$rank}){
	    $int11ptAP += $int_prec;
	    $int11ptP[$current_cut] += $int_prec;
	    $current_cut--;
	}

    }

    while ($current_cut >= 0){
	$int11ptAP += $int_prec;
	$int11ptP[$current_cut] += $int_prec;
	$current_cut--;
    }

    $int11ptAP /=11;
    $sum_int11ptAP += $int11ptAP;
    if ($print_all_queries){
	    #printf "int11ptAP\t\t$topic\t\t%6.4f\n", $int11ptAP;
	    }
}


$mean_avg_prec = $sum_avg_prec/$num_topics;
$mean_ndcg = $sum_ndcg/$num_topics;
$mean_int11ptAP = $sum_int11ptAP/$num_topics;

printf "infAP\t\tall\t\t%6.4f\n", $mean_avg_prec;
#printf "%6.3f\n", $mean_int11ptAP;
printf "infNDCG\t\tall\t\t%6.4f\n", $mean_ndcg;

for ($i=0;$i<=10;$i++){
    $tmp = $int11ptP[$i]/$num_topics;
    
    printf "iprec\@rec%4.2f\tall\t\t%6.4f\n", $i/10, $tmp;
}

# Now print estimated precisions
for ($i=0;$i<=$#precisionranks;$i++){
    $cutoff = $precisionranks[$i];
    $prec = $meanprecs[$i]/$num_topics;
    printf "iP$cutoff\t\tall\t\t%6.4f\n", $prec;
}

printf "inum_rel_ret\tall\t%14.4f\n", $sum_rel_ret;
printf "inum_rel\tall\t%14.4f\n", $sum_num_rel;
printf "num_ret\t\tall\t%9d\n", $sum_num_ret;
 

 
